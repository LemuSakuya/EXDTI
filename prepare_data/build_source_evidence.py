"""Build source-resolved DTI records without treating missing actions as negatives.

The default input is the local DrugBank action inventory. Its row is the source
unit; cited PMIDs remain unverified reference metadata, never extra sources.
Partitions group ID pairs and exact molecular/sequence aliases before splitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from exdti_evidence.provenance import attach_channels, revision_census

ACTIONS = ("inhibitor", "antagonist", "agonist")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_sequence(text):
    return "".join(line.strip() for line in (text or "").splitlines()
                   if not line.lstrip().startswith(">") and not line.lstrip().startswith(";")).replace(" ", "").upper()


CONDITION_FIELDS = {  # patent claim 3 / [0071]: species, target subtype, experimental condition
    "organism": ("organism", "species", "target_organism"),
    "target_subtype": ("target_subtype", "subtype", "isoform"),
    "assay_context": ("assay_context", "assay", "experimental_condition"),
}


def normalized_condition(row):
    """Normalized key-value condition; missing fields stay explicitly unknown."""
    condition = {"scope": "database_assertion", "assay_context": "unspecified"}
    for key, candidates in CONDITION_FIELDS.items():
        for name in candidates:
            value = row.get(name)
            if isinstance(value, str):
                value = " ".join(value.split())
            if value not in (None, "", [], {}):
                condition[key] = value
                break
    return dict(sorted(condition.items()))


def drugbank_record(row):
    drug_id, target_id = row["drugbank_id"], row["target_id"]
    release = row.get("source_version", "unspecified-local-export")
    source_id = "drugbank:" + digest(release)[:12] + ":" + row["record_id"]
    condition = normalized_condition(row)
    actions = sorted(set(str(a).strip().lower() for a in row.get("actions_raw", []) if str(a).strip()))
    assertions = [dict(drug_id=drug_id, target_id=target_id, action=action,
                       polarity=1, condition=condition.copy()) for action in actions]
    # Text describes exactly the database annotation, not the referenced paper.
    source_text = (f"Database annotation for drug {drug_id} and target {target_id}: "
                   + ("; ".join(actions) if actions else "action annotation unavailable") + ".")
    record = {
        "record_id": row["record_id"],
        "drug": {"id": drug_id, "smiles": row["smiles_raw"].strip()},
        "target": {"id": target_id, "sequence": clean_sequence(row.get("sequence_no_header") or row["sequence_raw"])},
        "action_label": row.get("core_label") if row.get("action_status") == "core_single" else None,
        "sources": [{"source_id": source_id, "text": source_text, "kind": "database_assertion",
                     "assertions": assertions, "metadata": {
                         "database": "DrugBank", "release": release, "record_id": row["record_id"],
                         "pmids_unverified": row.get("target_pmids", []),
                         "source_unit": "one exported target relation row",
                         "literature_verified": False,
                         "original_entities": [drug_id, target_id]}}],
        "claims": [dict(claim_id=f"action-{a}", drug_id=drug_id, target_id=target_id,
                        action=a, condition=condition.copy()) for a in ACTIONS],
        "metadata": {"action_status": row.get("action_status"), "actions_raw": actions,
                     "drug_level_moa_used_as_pair_evidence": False,
                     "input_record_ids": [row["record_id"]], "dataset": "drugbank"},
    }
    return attach_channels(record)


def identity_groups(records, canonical_smiles=False):
    """Union ID-pair aliases and exact structure+sequence aliases transitively.

    Default retains stereochemistry/salts and does not equate structural similarity
    with identity. Optional RDKit canonicalization normalizes notation only.
    """
    canon = lambda x: x.strip()
    if canonical_smiles:
        from rdkit import Chem

        def canon(text):
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError("Invalid SMILES: canonical identity cannot be established")
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[b] = a

    seen = {}
    for index, row in enumerate(records):
        sequence = clean_sequence(row["target"]["sequence"])
        row["target"]["sequence"] = sequence
        keys = ["id:" + row["drug"]["id"] + ":" + row["target"]["id"],
                "structure:" + digest(canon(row["drug"]["smiles"]) + "\0" + sequence)]
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row)
    return list(groups.values())


def merge_group(group):
    representative = json.loads(json.dumps(sorted(group, key=lambda r: r["record_id"])[0]))
    input_ids = sorted({i for row in group for i in row.get("metadata", {}).get("input_record_ids", [row["record_id"]])})
    representative["record_id"] = "pair-" + digest("\0".join(input_ids))[:20]
    representative.setdefault("metadata", {})["input_record_ids"] = input_ids
    representative["metadata"]["id_aliases"] = sorted({(r["drug"]["id"], r["target"]["id"]) for r in group})
    labels = sorted({r["action_label"] for r in group if r.get("action_label") in ACTIONS})
    # A single-action row may have an alias whose multi-action annotation adds a
    # second core action. Such a merged pair is not valid single-label training.
    if labels:
        labels = sorted(set(labels) | {a["action"] for row in group for source in row["sources"]
                                      for a in source.get("assertions", []) if a["action"] in ACTIONS and a.get("polarity") == 1})
    representative["metadata"]["observed_core_labels"] = labels
    representative["metadata"]["label_ambiguous"] = len(labels) > 1
    representative["metadata"]["alias_rows_are_not_independent_publications"] = True
    representative["action_label"] = labels[0] if len(labels) == 1 else None
    sources, claims = {}, {}
    for row in group:
        for src in row["sources"]:
            src = json.loads(json.dumps(src))
            for assertion in src.get("assertions", []):
                assertion["drug_id"] = representative["drug"]["id"]
                assertion["target_id"] = representative["target"]["id"]
            sources[src["source_id"]] = src
        for claim in row["claims"]:
            claim = dict(claim, drug_id=representative["drug"]["id"], target_id=representative["target"]["id"])
            key = json.dumps([claim["action"], claim.get("condition", {})], sort_keys=True)
            claim["claim_id"] = "claim-" + digest(key)[:12]
            claims[key] = claim
    representative["sources"] = sorted(sources.values(), key=lambda s: s["source_id"])
    representative["claims"] = [claims[key] for key in sorted(claims)]
    return attach_channels(representative)


def partition(records, seed=421):
    """Deterministic class-stratified grouped split; all views inherit this split."""
    bins, assigned = defaultdict(list), {}
    for row in records:
        key = "quarantine" if row["metadata"]["label_ambiguous"] else row.get("action_label") or "unlabeled"
        bins[key].append(row)
    rng = random.Random(seed)
    for label in sorted(bins):
        rows = sorted(bins[label], key=lambda r: r["record_id"])
        rng.shuffle(rows)
        ndev, ntest = int(len(rows) * 0.1), int(len(rows) * 0.1)
        for index, row in enumerate(rows):
            split = "dev" if index < ndev else "test" if index < ndev + ntest else "train"
            row["split"] = split
            assigned[row["record_id"]] = split
    return assigned


def build(input_path, output_dir, input_format="drugbank", seed=421, canonical_smiles=False):
    output_dir = Path(output_dir)
    raw = read_jsonl(input_path)
    records = [drugbank_record(row) for row in raw] if input_format == "drugbank" else raw
    for row in records:
        if not row["drug"].get("smiles") or not row["target"].get("sequence"):
            raise ValueError(f"Missing structure/sequence in {row['record_id']}")
    merged = sorted((merge_group(group) for group in identity_groups(records, canonical_smiles)), key=lambda r:r["record_id"])
    assigned = partition(merged, seed)
    core = [r for r in merged if r["action_label"] in ACTIONS and not r["metadata"]["label_ambiguous"]]
    quarantine = [r for r in merged if r["metadata"]["label_ambiguous"]]
    unlabeled = [r for r in merged if r["action_label"] is None and not r["metadata"]["label_ambiguous"]]
    write_jsonl(output_dir / "all_records.jsonl", merged)
    write_jsonl(output_dir / "unlabeled.jsonl", unlabeled)
    write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    for split in ("train", "dev", "test"):
        selected = [r for r in core if r["split"] == split]
        write_jsonl(output_dir / f"{split}.jsonl", selected)
        # Views reference their parent; no copied gold state enters a model input.
        interventions = [dict(view_id=r["record_id"] + ":remove:" + src["source_id"],
                              record_id=r["record_id"], split=split,
                              removed_source_ids=[src["source_id"]],
                              intervention_scope="database_row", synthetic=False)
                         for r in selected for src in r["sources"]]
        write_jsonl(output_dir / "interventions" / f"{split}.jsonl", interventions)
    manifest = {"seed": seed, "split_mode": "pair_and_exact_structure_sequence_grouped",
                "fractions": {"train": .8, "dev": .1, "test": .1},
                "canonical_smiles": canonical_smiles,
                "assignments": assigned,
                "input_record_assignments": {i:r["split"] for r in merged for i in r["metadata"]["input_record_ids"]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "input_path": str(Path(input_path).resolve()), "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "input_rows": len(raw), "grouped_pairs": len(merged), "eligible_core_pairs": len(core),
        "unlabeled_or_other_pairs": len(unlabeled), "quarantined_label_ambiguous_pairs": len(quarantine),
        "input_action_status_counts": dict(Counter(r.get("action_status", "adapter") for r in raw)),
        "core_action_counts": dict(Counter(r["action_label"] for r in core)),
        "split_core_counts": {s:dict(Counter(r["action_label"] for r in core if r["split"] == s)) for s in ("train", "dev", "test")},
        "split_total_counts": dict(Counter(r["split"] for r in core)),
        "core_pairs_with_multiple_database_rows": sum(len(r["sources"]) > 1 for r in core),
        "core_database_row_sources": sum(len(r["sources"]) for r in core),
        "identity": "RDKit isomeric canonical SMILES + header-stripped sequence" if canonical_smiles else "exact raw SMILES + header-stripped sequence, linked by ID pairs",
        "scope": "Database annotation faithfulness. Not verified literature explanations or independent DTI discovery.",
        "limitations": ["Random grouped-pair split is not a cold-drug/cold-target split.",
                        "A referenced PMID is not an independent verified supporting assertion.",
                        "Missing action is neither a negative interaction nor an agonist label.",
                        "Antagonist and inhibitor annotations are not logically opposite assertions.",
                        "Duplicate database rows need not be independent evidence.",
                        "Drug-level mechanism-of-action text is not used as pair-specific evidence."]}
    if core and core[0].get("metadata", {}).get("dataset") == "chembl":
        audit["limitations"] = [
            "Source text is the curated mechanism_of_action on that mechanism row.",
            "Rows are the first API results for each action type, not a random or complete extract.",
            "Shared target sequences with DrugBank do not make this an unseen-target benchmark.",
            "Each accepted pair has one mechanism source, so alternative-citation revisions do not arise.",
            "Reference identifiers stored on the mechanism row are not separate verified sources.",
        ]
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def materialize_tree(source_dir, output_dir):
    """Copy a frozen evidence tree and attach text/graph/cache channels.

    Split assignment, source identifiers, and intervention lists stay as they
    are. Records that are not drug-target evidence are copied unchanged.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("Refusing to overwrite the source tree; choose a new output directory")
    copied = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        target = output_dir / path.relative_to(source_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            rows = read_jsonl(path)
            write_jsonl(target, [attach_channels(row) if isinstance(row, dict) and {"drug", "target", "sources", "claims"} <= row.keys() else row for row in rows])
        else:
            target.write_bytes(path.read_bytes())
        copied += 1
    census = {}
    for split in ("train", "dev", "test"):
        path = output_dir / f"{split}.jsonl"
        if path.exists():
            census[split] = revision_census(read_jsonl(path))
    if census:
        (output_dir / "revision_census.json").write_text(json.dumps(census, indent=2), encoding="utf-8")
    return {"files": copied, "revision_census": census}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/action_v1/all_eligible.jsonl")
    parser.add_argument("--output", default="data/source_evidence/drugbank")
    parser.add_argument("--format", choices=("drugbank", "evidence"), default="drugbank")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--canonical-smiles", action="store_true", help="Require RDKit; retain salts and stereochemistry")
    parser.add_argument("--materialize-from", help="Copy this frozen evidence directory and attach channels")
    args = parser.parse_args()
    if args.materialize_from:
        print(json.dumps(materialize_tree(args.materialize_from, args.output), ensure_ascii=False, indent=2))
        return
    print(json.dumps(build(args.input, args.output, args.format, args.seed, args.canonical_smiles), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
