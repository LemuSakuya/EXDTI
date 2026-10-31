"""Validated, JSON-serializable records for provenance-aware DTI experiments.

The schema separates observations (sources and their extracted assertions),
queries (claims), and supervision (computed elsewhere).  Missing action labels
remain missing; an unlabelled pair is never converted to a negative example.
"""

from copy import deepcopy
import json


STATE_NAMES = ("insufficient", "supported", "refuted", "conflict")
CHANNELS = ("text", "graph", "cache")


class SchemaError(ValueError):
    """A record cannot be interpreted without an ambiguous assumption."""


def _identifier(value, location):
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{location} must be a nonempty string")
    return value


def condition_key(condition):
    """Exact, order-independent condition identity; absent fields are not wildcards."""
    if not isinstance(condition, dict):
        raise SchemaError("condition must be a dictionary")
    try:
        return json.dumps(condition, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError("condition must contain finite JSON values") from exc


def assertion_key(assertion):
    return (assertion["drug_id"], assertion["target_id"], assertion["action"],
            condition_key(assertion.get("condition", {})))


def _validate_assertion(assertion, location, is_claim=False):
    if not isinstance(assertion, dict):
        raise SchemaError(f"{location} must be an object")
    for field in ("drug_id", "target_id", "action"):
        _identifier(assertion.get(field), f"{location}.{field}")
    assertion.setdefault("condition", {})
    condition_key(assertion["condition"])
    if is_claim:
        _identifier(assertion.get("claim_id"), f"{location}.claim_id")
    elif type(assertion.get("polarity")) is not int or assertion["polarity"] not in (-1, 1):
        raise SchemaError(f"{location}.polarity must be the integer 1 or -1")


def validate_record(record):
    """Return a normalized copy or raise SchemaError, without editing the caller."""
    if not isinstance(record, dict):
        raise SchemaError("record must be an object")
    result = deepcopy(record)
    _identifier(result.get("record_id"), "record_id")
    for entity, representation in (("drug", "smiles"), ("target", "sequence")):
        if not isinstance(result.get(entity), dict):
            raise SchemaError(f"{entity} must be an object")
        _identifier(result[entity].get("id"), f"{entity}.id")
        _identifier(result[entity].get(representation), f"{entity}.{representation}")
    if result.get("action_label") is not None and (
        isinstance(result["action_label"], bool)
        or not isinstance(result["action_label"], (str, int))
    ):
        raise SchemaError("action_label must be a string, integer, or null")
    for name in ("sources", "claims"):
        if not isinstance(result.get(name), list):
            raise SchemaError(f"{name} must be a list (possibly empty)")
    source_ids = set()
    for index, source in enumerate(result["sources"]):
        prefix = f"sources[{index}]"
        if not isinstance(source, dict):
            raise SchemaError(f"{prefix} must be an object")
        source_id = _identifier(source.get("source_id"), prefix + ".source_id")
        if source_id in source_ids:
            raise SchemaError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        source.setdefault("source_unit_id", source_id)
        _identifier(source["source_unit_id"], prefix + ".source_unit_id")
        source.setdefault("kind", "unspecified")
        _identifier(source["kind"], prefix + ".kind")
        source.setdefault("text", "")
        if not isinstance(source["text"], str):
            raise SchemaError(f"{prefix}.text must be a string")
        source.setdefault("assertions", [])
        if not isinstance(source["assertions"], list):
            raise SchemaError(f"{prefix}.assertions must be a list")
        for position, assertion in enumerate(source["assertions"]):
            _validate_assertion(assertion, f"{prefix}.assertions[{position}]")
    claim_ids = set()
    for index, claim in enumerate(result["claims"]):
        _validate_assertion(claim, f"claims[{index}]", is_claim=True)
        if claim["claim_id"] in claim_ids:
            raise SchemaError(f"duplicate claim_id: {claim['claim_id']}")
        claim_ids.add(claim["claim_id"])
        if (claim["drug_id"], claim["target_id"]) != (result["drug"]["id"], result["target"]["id"]):
            raise SchemaError("claim entity IDs must match the directed drug-target pair")
    if "artifacts" in result and "derived_artifacts" in result:
        raise SchemaError("use artifacts or derived_artifacts, not both")
    result["artifacts"] = result.pop("derived_artifacts", result.get("artifacts", []))
    if not isinstance(result["artifacts"], list):
        raise SchemaError("artifacts must be a list")
    artifact_ids = set()
    for index, artifact in enumerate(result["artifacts"]):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            raise SchemaError(f"{prefix} must be an object")
        artifact_id = _identifier(artifact.get("artifact_id"), prefix + ".artifact_id")
        if artifact_id in artifact_ids:
            raise SchemaError(f"duplicate artifact_id: {artifact_id}")
        artifact_ids.add(artifact_id)
        if artifact.get("channel") not in CHANNELS:
            raise SchemaError(f"{prefix}.channel must be one of {CHANNELS}")
        groups = artifact.setdefault("support_sets", [])
        parents = artifact.setdefault("depends_on_artifact_ids", [])
        if not isinstance(groups, list) or not isinstance(parents, list):
            raise SchemaError(f"{prefix} dependency fields must be lists")
        if not groups and not parents:
            raise SchemaError(f"{prefix} must have source or artifact dependencies")
        for group in groups:
            if not isinstance(group, list) or not group:
                raise SchemaError(f"{prefix} support alternatives must be nonempty lists")
            for source_id in group:
                _identifier(source_id, prefix + ".support_sets")
                if source_id not in source_ids:
                    raise SchemaError(f"unknown source dependency {source_id}")
        for parent_id in parents:
            _identifier(parent_id, prefix + ".depends_on_artifact_ids")
        artifact.setdefault("assertions", [])
        if not isinstance(artifact["assertions"], list):
            raise SchemaError(f"{prefix}.assertions must be a list")
        for position, assertion in enumerate(artifact["assertions"]):
            _validate_assertion(assertion, f"{prefix}.assertions[{position}]")
    for artifact in result["artifacts"]:
        unknown = set(artifact["depends_on_artifact_ids"]) - artifact_ids
        if unknown:
            raise SchemaError(f"unknown artifact dependencies: {sorted(unknown)}")
    return result
