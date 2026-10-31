"""The fixtures are synthetic provenance examples, never pharmacological results."""

from copy import deepcopy
import unittest

from exdti_evidence.metrics import evaluate_claims, evaluate_revision
from exdti_evidence.provenance import assess_claims, attach_channels, build_evidence, intervention_pair, render_claims, revision_census
from exdti_evidence.schema import SchemaError, validate_record


def assertion(action="inhibitor", polarity=1, condition=None):
    return {"drug_id": "D", "target_id": "T", "action": action,
            "polarity": polarity, "condition": {} if condition is None else condition}


def fixture():
    return {"record_id": "synthetic-1", "drug": {"id": "D", "smiles": "CCO"},
            "target": {"id": "T", "sequence": "MGL"}, "action_label": None,
            "sources": [
                {"source_id": "s1", "kind": "synthetic", "text": "Only source one marker.",
                 "assertions": [assertion("a"), assertion("b")]},
                {"source_id": "s2", "kind": "synthetic", "text": "Source two marker.",
                 "assertions": [assertion("b")]},
                {"source_id": "s3", "kind": "synthetic", "text": "Source three marker.",
                 "assertions": [assertion("c")]},
            ],
            "claims": [{"claim_id": "c" + str(i), "drug_id": "D", "target_id": "T",
                        "action": action, "condition": {}}
                       for i, action in enumerate(("a", "b", "c"), 1)]}


class ProvenanceTests(unittest.TestCase):
    def test_diagram_local_revision_and_citations(self):
        record = fixture()
        pair = intervention_pair(record, ["s1"])
        self.assertEqual([row["state"] for row in pair["after"]["claims"]],
                         ["insufficient", "supported", "supported"])
        self.assertEqual([row["revision"] for row in pair["transitions"]],
                         ["change_state", "update_citation", "retain"])
        self.assertEqual(pair["after"]["claims"][1]["cited_source_ids"], ["s2"])
        self.assertNotIn("Sources: [s1]", render_claims(record, pair["after"]["claims"]))

    def test_joint_and_alternative_derived_support(self):
        record = fixture()
        record["sources"][0]["assertions"] = []
        record["artifacts"] = [{"artifact_id": "rule", "channel": "graph",
                                "support_sets": [["s1", "s2"], ["s3"]],
                                "assertions": [assertion("a")]}]
        before = assess_claims(record)[0]
        self.assertEqual(before["support_sets"], [["s3"], ["s1", "s2"]])
        self.assertEqual(assess_claims(record, build_evidence(record, ["s1"]))[0]["state"], "supported")
        after = assess_claims(record, build_evidence(record, ["s1", "s3"]))[0]
        self.assertEqual(after["state"], "insufficient")
        joint_only = assess_claims(record, build_evidence(record, ["s3"]))[0]
        self.assertEqual(joint_only["cited_source_ids"], ["s1", "s2"])

    def test_second_hop_cache_rebuilt_and_tainted_content_removed(self):
        record = fixture()
        record["artifacts"] = [
            {"artifact_id": "text1", "channel": "text", "support_sets": [["s1"], ["s2"]],
             "payload": {"text": "STALE secret cache from s1"}},
            {"artifact_id": "graph2", "channel": "graph", "depends_on_artifact_ids": ["text1"],
             "support_sets": [["s3"]]},
            {"artifact_id": "cache3", "channel": "cache", "depends_on_artifact_ids": ["graph2"],
             "payload": {"text": "STALE secret cache from s1"}},
        ]
        before = build_evidence(record)
        after = build_evidence(record, ["s1"])
        self.assertEqual(len(after["artifacts"]), 3)
        cache = {row["artifact_id"]: row for row in after["artifacts"]}["cache3"]
        self.assertEqual(cache["support_sets"], [["s2", "s3"]])
        self.assertNotIn("Only source one marker", cache["payload"]["text"])
        self.assertNotIn("STALE", str(after["artifacts"]))
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        dead = build_evidence(record, ["s1", "s2"])
        self.assertEqual(dead["artifacts"], [])

    def test_conflict_requires_same_action_and_exact_conditions(self):
        record = fixture()
        record["sources"][1]["assertions"] = [assertion("a", -1, {"species": "mouse"}),
                                                 assertion("opposite_action", 1)]
        self.assertEqual(assess_claims(record)[0]["state"], "supported")
        record["sources"][1]["assertions"].append(assertion("a", -1))
        self.assertEqual(assess_claims(record)[0]["state"], "conflict")
        self.assertEqual(assess_claims(record)[0]["cited_source_ids"], ["s1", "s2"])
        self.assertEqual(assess_claims(record, build_evidence(record, ["s1"]))[0]["state"], "refuted")

    def test_source_unit_aliases_withdraw_together(self):
        record = fixture()
        record["sources"][0]["source_unit_id"] = "database-assertion-1"
        record["sources"][1]["source_unit_id"] = "database-assertion-1"
        record["sources"][0]["publication_ids"] = ["PMID:123"]
        result = build_evidence(record, ["s1"])
        self.assertEqual(result["removed_source_ids"], ["s1", "s2"])
        self.assertEqual(result["source_ids"], ["s3"])
        self.assertNotIn("PMID:123", assess_claims(record)[0]["cited_source_ids"])

    def test_schema_missing_label_is_not_a_negative(self):
        record = fixture()
        original = deepcopy(record)
        normalized = validate_record(record)
        self.assertIsNone(normalized["action_label"])
        del record["action_label"]
        self.assertNotIn("action_label", validate_record(record))
        self.assertEqual(original["sources"][0].get("source_unit_id"), None)
        with self.assertRaises(SchemaError):
            build_evidence(record, ["not-a-source"])

    def test_cycle_and_unknown_dependencies_rejected(self):
        record = fixture()
        record["artifacts"] = [{"artifact_id": "a", "channel": "cache",
                                "depends_on_artifact_ids": ["b"]},
                               {"artifact_id": "b", "channel": "cache",
                                "depends_on_artifact_ids": ["a"]}]
        with self.assertRaises(SchemaError):
            build_evidence(record)
        record["artifacts"][0]["depends_on_artifact_ids"] = ["unknown"]
        with self.assertRaises(SchemaError):
            build_evidence(record)

    def test_metrics_count_coverage_retention_and_invalid_citations(self):
        pair = intervention_pair(fixture(), ["s1"])
        before, after = pair["before"]["claims"], pair["after"]["claims"]
        good = evaluate_claims(after, after, ["s2", "s3"])
        self.assertEqual(good["coverage"], {"numerator": 2, "denominator": 3, "value": 2 / 3})
        self.assertEqual(good["unsupported_claim_rate"]["numerator"], 0)
        wrong = deepcopy(before)
        bad = evaluate_claims(after, wrong, ["s2", "s3"])
        self.assertEqual(bad["unsupported_claim_rate"]["numerator"], 1)
        self.assertGreater(bad["withdrawn_or_unknown_citation_rate"]["numerator"], 0)
        revision = evaluate_revision(before, after, before, after)
        self.assertEqual(revision["supported_retention"]["value"], 1)
        self.assertEqual(revision["required_state_change_success"]["value"], 1)
        empty = evaluate_claims(after, [], ["s2", "s3"])
        self.assertIsNone(empty["unsupported_claim_rate"]["value"])
        self.assertEqual(empty["coverage"]["value"], 0)

    def test_attached_channels_leave_with_their_source(self):
        record = attach_channels(fixture())
        self.assertEqual(len(record["artifacts"]), 9)
        self.assertEqual([row["channel"] for row in record["artifacts"][:3]], ["text", "graph", "cache"])
        after = build_evidence(record, ["s1"])
        surviving = {row["artifact_id"] for row in after["artifacts"]}
        self.assertTrue({"text:s2", "graph:s2", "cache:s2"} <= surviving)
        self.assertFalse(any(artifact_id.endswith(":s1") for artifact_id in surviving))
        cache = next(row for row in after["artifacts"] if row["artifact_id"] == "cache:s2")
        self.assertIn("Source two", cache["payload"]["text"])
        self.assertNotIn("Source one", cache["payload"]["text"])
        self.assertNotIn("STALE", str(after["artifacts"]))
        census = revision_census([record])
        self.assertEqual(census["interventions"], 3)
        self.assertGreater(census["claims_change_state"], 0)
        self.assertGreater(census["claims_retain"], 0)
        self.assertEqual(census["claims_touching_conflict"], 0)

    def test_joint_witness_requires_every_member(self):
        ref = [{"claim_id": "x", "state": "supported", "support_sets": [["s1", "s2"]],
                "refutation_sets": [], "supporting_source_ids": ["s1", "s2"],
                "refuting_source_ids": []}]
        pred = [{"claim_id": "x", "state": "supported", "cited_source_ids": ["s1"]}]
        metrics = evaluate_claims(ref, pred)
        self.assertEqual(metrics["citation_precision"]["value"], 1)
        self.assertEqual(metrics["citation_witness_completeness"]["value"], 0)


if __name__ == "__main__":
    unittest.main()
