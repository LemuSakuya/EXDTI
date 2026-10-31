"""Export the user's three-source example. Synthetic mechanism test, not biomedical evidence."""
import argparse
import json
from pathlib import Path
from .provenance import intervention_pair, render_claims


def diagram_example():
    claims = [{"claim_id": "c" + str(i), "drug_id": "D", "target_id": "T", "action": action,
               "condition": {"synthetic_assay": str(i)}}
              for i, action in enumerate(("inhibitor", "antagonist", "agonist"), 1)]
    sources = []
    for source_id, indices in (("s1", (0, 1)), ("s2", (1,)), ("s3", (2,))):
        assertions = [{**{k: v for k, v in claims[i].items() if k != "claim_id"}, "polarity": 1} for i in indices]
        sources.append({"source_id": source_id, "kind": "synthetic_fixture",
                        "text": "Synthetic source " + source_id + " for " + ", ".join(claims[i]["claim_id"] for i in indices),
                        "assertions": assertions})
    return {"record_id": "synthetic-user-diagram", "drug": {"id": "D", "smiles": "CCO"},
            "target": {"id": "T", "sequence": "ACDE"}, "action_label": None,
            "metadata": {"synthetic": True, "purpose": "diagram dependency semantics; no real pharmacology claim"},
            "claims": claims, "sources": sources,
            "artifacts": [{"artifact_id": "cache-" + s["source_id"], "channel": "cache",
                           "support_sets": [[s["source_id"]]], "payload": "stale input intentionally ignored"} for s in sources]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/source_intervention_demo/diagram.json")
    args = parser.parse_args()
    record = diagram_example()
    pair = intervention_pair(record, ["s1"])
    result = {"synthetic": True, "record": record, "intervention": pair,
              "before_text": render_claims(record, pair["before"]["claims"]),
              "after_text": render_claims(record, pair["after"]["claims"])}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pair["transitions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
