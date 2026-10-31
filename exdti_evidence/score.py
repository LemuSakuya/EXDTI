"""Score saved structured claims; raw generated prose is not auto-certified."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from .metrics import evaluate_claims, evaluate_revision


def pooled(rows, side, threshold=0.0, baseline=False):
    reference, predictions, available = [], [], set()
    for row in rows:
        prefix = row["record_id"] + "::"
        gold = deepcopy(row[side]["oracle_claims_for_scoring_only"])
        pred = deepcopy(gold if baseline else row[side]["claims"])
        for item in gold + pred:
            item["claim_id"] = prefix + item["claim_id"]
            for key in ("supporting_source_ids", "refuting_source_ids", "cited_source_ids"):
                if key in item:
                    item[key] = [prefix + s for s in item[key]]
            for key in ("support_sets", "refutation_sets"):
                if key in item:
                    item[key] = [[prefix + s for s in group] for group in item[key]]
        for item in pred:
            if not baseline and max(item["state_probabilities"]) < threshold:
                item["state"], item["cited_source_ids"] = "insufficient", []
        reference.extend(gold); predictions.extend(pred)
        available.update(prefix + s for s in row[side]["available_source_ids"])
    return reference, predictions, available


def score(rows, threshold=0.0, baseline=False):
    old, before, old_sources = pooled(rows, "before", threshold, baseline)
    new, after, new_sources = pooled(rows, "after", threshold, baseline)
    return {"confidence_threshold": threshold,
            "before": evaluate_claims(old, before, sorted(old_sources)),
            "after": evaluate_claims(new, after, sorted(new_sources)),
            "revision": evaluate_revision(old, new, before, after)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line]
    result = {"records": len(rows), "model": score(rows), "symbolic_rules_baseline": score(rows, baseline=True),
              "risk_coverage_curve": [score(rows, threshold=t) for t in (0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)],
              "scope": "Database assertion / structured planner assessment only; raw prose semantics are not evaluated.",
              "baseline_note": "Exact rules use observable assertions and are also the database-label oracle. Perfect rules scores here are construction checks, not independent empirical evidence.",
              "threshold_policy": "Choose operating threshold on development data; compare systems at overlapping coverage. Test curves are descriptive, not a threshold-selection procedure."}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(rows), "output": args.output}))


if __name__ == "__main__":
    main()
