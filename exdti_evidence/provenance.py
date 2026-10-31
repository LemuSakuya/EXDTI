"""Source-synchronized interventions and an explicit symbolic baseline.

This module assesses database/extracted assertions, not biological truth.  A
negative polarity means an explicit opposing assertion about the SAME action
under the SAME condition.  Different action names do not imply contradiction.
"""

from copy import deepcopy
import hashlib
import json

from .schema import SchemaError, assertion_key, condition_key, validate_record


def _canonical_groups(groups):
    """Deduplicate provenance DNF; remove supersets made redundant by alternatives."""
    unique = {frozenset(group) for group in groups}
    minimal = [group for group in unique if not any(other < group for other in unique)]
    return sorted((sorted(group) for group in minimal), key=lambda group: (len(group), group))


def _joint(left, right):
    return _canonical_groups([set(a) | set(b) for a in left for b in right])


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def build_evidence(record, removed_source_ids=()):
    """Rebuild observable text, graph, and cache views from surviving sources.

    ``support_sets`` is OR-of-AND provenance.  Parent artifact dependencies are
    conjoined, then expanded to source-level support alternatives.  Input cache
    payloads are NEVER used.  All returned artifacts are projections rebuilt
    from active sources and explicitly declared extracted graph assertions.

    Optional ``source_unit_id`` aliases represent copies of the same provenance
    unit.  Withdrawing any alias withdraws the whole unit.  A database row that
    merely lists a PMID is still a database source, not a read publication.
    """
    record = validate_record(record)
    requested = set(removed_source_ids)
    sources = {source["source_id"]: source for source in record["sources"]}
    unknown = requested - sources.keys()
    if unknown:
        raise SchemaError(f"cannot withdraw unknown source IDs: {sorted(unknown)}")
    removed_units = {sources[source_id]["source_unit_id"] for source_id in requested}
    removed = {source_id for source_id, source in sources.items()
               if source["source_unit_id"] in removed_units}
    active = {source_id: source for source_id, source in sources.items() if source_id not in removed}
    definitions = {artifact["artifact_id"]: artifact for artifact in record["artifacts"]}
    rebuilt = {}
    visited = set()

    def rebuild(artifact_id):
        if artifact_id in rebuilt:
            return rebuilt[artifact_id]
        if artifact_id in visited:
            raise SchemaError(f"cyclic artifact dependency involving {artifact_id}")
        visited.add(artifact_id)
        definition = definitions[artifact_id]
        groups = ([group for group in definition["support_sets"] if set(group) <= active.keys()]
                  if definition["support_sets"] else [[]])
        for parent_id in definition["depends_on_artifact_ids"]:
            parent = rebuild(parent_id)
            groups = _joint(groups, parent["support_sets"] if parent is not None else [])
        groups = _canonical_groups(groups)
        visited.remove(artifact_id)
        if not groups:
            rebuilt[artifact_id] = None
            return None
        source_ids = sorted({source_id for group in groups for source_id in group})
        assertions = []
        for source_id in source_ids:
            for assertion in active[source_id]["assertions"]:
                assertions.append(dict(deepcopy(assertion), support_sets=[[source_id]]))
        for assertion in definition["assertions"]:
            assertions.append(dict(deepcopy(assertion), support_sets=deepcopy(groups)))
        # Dependencies may carry extracted facts not repeated in the child.
        for parent_id in definition["depends_on_artifact_ids"]:
            parent = rebuilt[parent_id]
            if parent is not None:
                assertions.extend(deepcopy(parent["payload"]["assertions"]))
        payload = {
            "source_ids": source_ids,
            "text": "\n".join(active[source_id]["text"] for source_id in source_ids
                              if active[source_id]["text"]),
            "assertions": assertions,
        }
        artifact = {"artifact_id": artifact_id, "channel": definition["channel"],
                    "support_sets": groups, "source_ids": source_ids,
                    "depends_on_artifact_ids": list(definition["depends_on_artifact_ids"]),
                    "payload": payload, "fingerprint": _fingerprint(payload)}
        rebuilt[artifact_id] = artifact
        return artifact

    for artifact_id in definitions:
        rebuild(artifact_id)
    graph_assertions = []
    for source_id, source in active.items():
        for assertion in source["assertions"]:
            graph_assertions.append(dict(deepcopy(assertion), support_sets=[[source_id]],
                                         origin="source", source_id=source_id))
    for artifact_id, artifact in rebuilt.items():
        if artifact is not None:
            for assertion in definitions[artifact_id]["assertions"]:
                graph_assertions.append(dict(deepcopy(assertion),
                                             support_sets=deepcopy(artifact["support_sets"]),
                                             origin="derived", artifact_id=artifact_id))
    return {
        "record_id": record["record_id"],
        "source_ids": sorted(active),
        "sources": [deepcopy(active[source_id]) for source_id in sorted(active)],
        "requested_removed_source_ids": sorted(requested),
        "removed_source_ids": sorted(removed),
        "artifacts": [rebuilt[key] for key in sorted(rebuilt) if rebuilt[key] is not None],
        "graph_assertions": graph_assertions,
        "fingerprint": _fingerprint({"sources": [active[key] for key in sorted(active)],
                                     "artifacts": [rebuilt[key] for key in sorted(rebuilt)
                                                   if rebuilt[key] is not None]}),
    }


def assess_claims(record, evidence=None):
    """Symbolic supervision/reference assessment; NEVER put this into model inputs."""
    record = validate_record(record)
    evidence = build_evidence(record) if evidence is None else evidence
    if evidence["record_id"] != record["record_id"]:
        raise SchemaError("evidence belongs to a different record")
    results = []
    for claim in record["claims"]:
        key = assertion_key(claim)
        positive, negative = [], []
        for assertion in evidence["graph_assertions"]:
            if assertion_key(assertion) != key:
                continue
            (positive if assertion["polarity"] == 1 else negative).extend(assertion["support_sets"])
        positive, negative = _canonical_groups(positive), _canonical_groups(negative)
        state = ("conflict" if positive and negative else "supported" if positive
                 else "refuted" if negative else "insufficient")
        positive_ids = sorted({source_id for group in positive for source_id in group})
        negative_ids = sorted({source_id for group in negative for source_id in group})
        # A shortest complete witness per polarity is enough, unlike a single
        # citation selected arbitrarily from a jointly sufficient support set.
        citations = sorted(set((positive[0] if positive else []) + (negative[0] if negative else [])))
        results.append({"claim_id": claim["claim_id"], "state": state,
                        "supporting_source_ids": positive_ids, "refuting_source_ids": negative_ids,
                        "support_sets": positive, "refutation_sets": negative,
                        "cited_source_ids": citations})
    return results


def attach_channels(record):
    """Give every database source a text, graph, and cache descendant.

    Source assertions remain the graph-channel observations. The cache artifact
    has no support of its own: it is rebuilt from the graph artifact, and any
    payload stored on input is discarded by ``build_evidence``.
    """
    record = validate_record(record)
    artifacts = []
    for source in record["sources"]:
        source_id = source["source_id"]
        text_id, graph_id, cache_id = f"text:{source_id}", f"graph:{source_id}", f"cache:{source_id}"
        artifacts.extend([
            {"artifact_id": text_id, "channel": "text", "support_sets": [[source_id]], "assertions": []},
            {"artifact_id": graph_id, "channel": "graph", "support_sets": [[source_id]],
             "depends_on_artifact_ids": [text_id], "assertions": []},
            {"artifact_id": cache_id, "channel": "cache", "depends_on_artifact_ids": [graph_id], "assertions": []},
        ])
    record["artifacts"] = artifacts
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["evidence_channels"] = ["text", "graph", "cache"]
    metadata["cache_recomputed_from_sources"] = True
    record["metadata"] = metadata
    return validate_record(record)


def revision_census(records):
    """Count local revisions produced by withdrawing one source at a time."""
    summary = {
        "records": 0, "interventions": 0, "claims_change_state": 0, "claims_update_citation": 0,
        "claims_retain": 0, "claims_same_state_support_set_changed": 0, "claims_touching_conflict": 0,
        "interventions_with_state_change": 0, "interventions_with_witness_update": 0,
        "interventions_with_support_set_update": 0, "interventions_with_all_three_revision_kinds": 0,
        "interventions_with_only_retention": 0,
    }
    for record in records:
        summary["records"] += 1
        for source in record.get("sources", []):
            pair = intervention_pair(record, [source["source_id"]])
            summary["interventions"] += 1
            revisions = [row["revision"] for row in pair["transitions"]]
            kinds = set(revisions)
            summary["interventions_with_state_change"] += int("change_state" in kinds)
            summary["interventions_with_witness_update"] += int("update_citation" in kinds)
            summary["interventions_with_only_retention"] += int(kinds == {"retain"})
            summary["interventions_with_all_three_revision_kinds"] += int(
                {"change_state", "update_citation", "retain"} <= kinds)
            support_changed = False
            for first, second, transition in zip(pair["before"]["claims"], pair["after"]["claims"], pair["transitions"]):
                summary["claims_" + transition["revision"]] += 1
                before_ids = set(first["supporting_source_ids"] + first["refuting_source_ids"])
                after_ids = set(second["supporting_source_ids"] + second["refuting_source_ids"])
                if first["state"] == second["state"] and before_ids != after_ids:
                    summary["claims_same_state_support_set_changed"] += 1
                    support_changed = True
                if "conflict" in (first["state"], second["state"]):
                    summary["claims_touching_conflict"] += 1
            summary["interventions_with_support_set_update"] += int(support_changed)
    return summary


def intervention_pair(record, removed_source_ids):
    """Make paired views and symbolic transition targets, without any prediction p."""
    before = build_evidence(record)
    after = build_evidence(record, removed_source_ids)
    old = assess_claims(record, before)
    new = assess_claims(record, after)
    transitions = []
    for first, second in zip(old, new):
        changed_state = first["state"] != second["state"]
        changed_citations = first["cited_source_ids"] != second["cited_source_ids"]
        transitions.append({"claim_id": first["claim_id"], "before_state": first["state"],
                            "after_state": second["state"], "state_changed": changed_state,
                            "citations_changed": changed_citations,
                            "revision": "change_state" if changed_state else
                            "update_citation" if changed_citations else "retain"})
    return {"before": {"evidence": before, "claims": old},
            "after": {"evidence": after, "claims": new}, "transitions": transitions}


def _has_complete_witness(groups, cited):
    return any(set(group) <= cited for group in groups)


def release_check(record, evidence, neural_claims):
    """Optional pre-release check (patent claim 9; description [0023], [0102]-[0103]).

    An independent source-witness verifier re-derives each claim state from the
    CURRENT active sources and explicit rules only (``assess_claims``): positive
    groups only -> supported, negative only -> refuted, both -> conflict,
    neither -> insufficient.  It first checks that every cited ID belongs to the
    active sources, then checks complete positive and negative witnesses
    separately.  A neural state or citation set without a matching complete
    witness is not released; the re-determined state and a valid witness are
    released with the rule template instead.  Refuted/conflict are NOT rewritten
    to insufficient; conflict lists both positive and negative sources.  The raw
    neural output is kept separately for audit.
    """
    active = set(evidence["source_ids"])
    verified = {item["claim_id"]: item for item in assess_claims(record, evidence)}
    released = []
    for claim in neural_claims:
        reference = verified[claim["claim_id"]]
        cited = set(claim.get("cited_source_ids", []))
        citations_active = cited <= active
        state = claim["state"]
        if state == "supported":
            witness = _has_complete_witness(reference["support_sets"], cited)
        elif state == "refuted":
            witness = _has_complete_witness(reference["refutation_sets"], cited)
        elif state == "conflict":
            witness = (_has_complete_witness(reference["support_sets"], cited)
                       and _has_complete_witness(reference["refutation_sets"], cited))
        else:
            witness = True
        passed = citations_active and witness and state == reference["state"]
        if passed:
            item = {"claim_id": claim["claim_id"], "state": state,
                    "cited_source_ids": sorted(cited), "release_origin": "neural_verified"}
        else:
            item = {"claim_id": claim["claim_id"], "state": reference["state"],
                    "cited_source_ids": list(reference["cited_source_ids"]),
                    "release_origin": "rule_template",
                    "template_text": render_claims(record, [reference])}
            if reference["state"] == "conflict":
                item["supporting_source_ids"] = reference["supporting_source_ids"]
                item["refuting_source_ids"] = reference["refuting_source_ids"]
        item["verification"] = {"passed": passed, "citations_in_active_sources": citations_active,
                                "complete_witness": witness, "state_matches_verifier": state == reference["state"]}
        released.append(item)
    return released


def render_claims(record, assessment):
    """Deterministic strong baseline. Wording reports source assertions, not truth."""
    record = validate_record(record)
    by_id = {claim["claim_id"]: claim for claim in record["claims"]}
    lines = []
    for result in assessment:
        claim = by_id[result["claim_id"]]
        condition = condition_key(claim["condition"])
        subject = f"{claim['drug_id']} / {claim['target_id']}: {claim['action']} (condition={condition})"
        citation = ", ".join(result.get("cited_source_ids", []))
        if result["state"] == "supported":
            statement = "The available source assertions support this claim."
        elif result["state"] == "refuted":
            statement = "The available source assertions oppose this claim."
        elif result["state"] == "conflict":
            statement = "The available source assertions disagree under the same recorded conditions."
        else:
            statement = "The available source assertions are insufficient for this claim."
        lines.append(f"{subject}. {statement}" + (f" Sources: [{citation}]." if citation else ""))
    return "\n".join(lines)
