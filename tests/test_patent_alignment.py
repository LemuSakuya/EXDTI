"""Patent-alignment tests (claims 3, 9, 10). Synthetic fixtures only."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare_data"))
from build_source_evidence import drugbank_record, normalized_condition
from exdti_evidence.provenance import build_evidence, release_check
from exdti_evidence.run import choose_removal
from test_provenance import assertion, fixture


def neural(claim_id, state, cited):
    return {"claim_id": claim_id, "state": state, "cited_source_ids": cited}


class ReleaseCheckTests(unittest.TestCase):
    def test_verified_output_is_released(self):
        record = fixture(); evidence = build_evidence(record)
        out = release_check(record, evidence, [neural("c3", "supported", ["s3"])])
        self.assertEqual(out[0]["release_origin"], "neural_verified")
        self.assertTrue(out[0]["verification"]["passed"])

    def test_withdrawn_citation_falls_back_to_verifier_state(self):
        record = fixture(); evidence = build_evidence(record, ["s1"])
        out = release_check(record, evidence, [neural("c1", "supported", ["s1"])])
        self.assertEqual(out[0]["state"], "insufficient")
        self.assertEqual(out[0]["cited_source_ids"], [])
        self.assertEqual(out[0]["release_origin"], "rule_template")
        self.assertFalse(out[0]["verification"]["citations_in_active_sources"])

    def test_missing_witness_uses_valid_witness(self):
        record = fixture(); evidence = build_evidence(record)
        out = release_check(record, evidence, [neural("c2", "supported", [])])
        self.assertEqual(out[0]["state"], "supported")
        self.assertTrue(set(out[0]["cited_source_ids"]) & {"s1", "s2"})
        self.assertIn("support this claim", out[0]["template_text"])

    def test_refuted_and_conflict_not_rewritten_to_insufficient(self):
        record = fixture()
        record["sources"][2]["assertions"].append(assertion("a", polarity=-1))
        evidence = build_evidence(record)
        out = release_check(record, evidence, [neural("c1", "insufficient", [])])
        self.assertEqual(out[0]["state"], "conflict")
        self.assertEqual(out[0]["supporting_source_ids"], ["s1"])
        self.assertEqual(out[0]["refuting_source_ids"], ["s3"])
        after = build_evidence(record, ["s1"])
        out = release_check(record, after, [neural("c1", "insufficient", [])])
        self.assertEqual(out[0]["state"], "refuted")

    def test_input_claims_are_not_mutated(self):
        record = fixture(); claims = [neural("c1", "supported", ["s9"])]
        release_check(record, build_evidence(record), claims)
        self.assertEqual(claims[0], neural("c1", "supported", ["s9"]))


class RemovalRuleTests(unittest.TestCase):
    def test_step_index_selects_one_source(self):
        record = fixture()
        self.assertEqual([choose_removal(record, i) for i in range(4)], [["s1"], ["s2"], ["s3"], ["s1"]])

    def test_explicit_interventions_take_precedence(self):
        record = fixture()
        record["interventions"] = [{"removed_source_ids": ["s2", "s3"]}, ["s1"]]
        self.assertEqual(choose_removal(record, 0), ["s2", "s3"])
        self.assertEqual(choose_removal(record, 1), ["s1"])

    def test_no_sources(self):
        record = fixture(); record["sources"] = []
        self.assertEqual(choose_removal(record, 0), [])


class ConditionTests(unittest.TestCase):
    def test_placeholder_kept_when_fields_absent(self):
        self.assertEqual(normalized_condition({}), {"assay_context": "unspecified", "scope": "database_assertion"})

    def test_fields_normalized(self):
        cond = normalized_condition({"species": "  Homo   sapiens ", "isoform": "CA7"})
        self.assertEqual(cond, {"assay_context": "unspecified", "organism": "Homo sapiens",
                                "scope": "database_assertion", "target_subtype": "CA7"})

    def test_drugbank_record_uses_condition(self):
        row = {"drugbank_id": "DB1", "target_id": "T1", "record_id": "r1", "smiles_raw": "CC",
               "sequence_raw": "ACDE", "actions_raw": ["inhibitor"], "organism": "Humans"}
        record = drugbank_record(row)
        self.assertEqual(record["sources"][0]["assertions"][0]["condition"]["organism"], "Humans")
        self.assertEqual(record["claims"][0]["condition"]["organism"], "Humans")


if __name__ == "__main__":
    unittest.main()
