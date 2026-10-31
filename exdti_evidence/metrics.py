"""Count-based metrics with explicit denominators for claim revision experiments."""

from .schema import STATE_NAMES


_POLARITIES = {"insufficient": set(), "supported": {1}, "refuted": {-1}, "conflict": {1, -1}}


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def evaluate_claims(reference, predicted, available_source_ids=None):
    """Missing outputs are abstentions; extra/duplicate claim IDs are invalid.

    Coverage = assertive outputs / all query claims. Unsupported rate = outputs
    asserting at least one polarity absent from reference / assertive outputs.
    Citation scores assess source provenance, not publication-level entailment.
    Exact citation witness completeness checks AND groups as well as IDs.
    """
    truth = {row["claim_id"]: row for row in reference}
    guesses = {row["claim_id"]: row for row in predicted}
    if len(truth) != len(reference) or len(guesses) != len(predicted):
        raise ValueError("duplicate claim IDs")
    if set(guesses) - set(truth):
        raise ValueError("predictions contain unknown claim IDs")
    states = {row["state"] for row in reference + predicted}
    if states - set(STATE_NAMES):
        raise ValueError("unknown claim state")
    available = None if available_source_ids is None else set(available_source_ids)
    correct = asserted = unsupported = valid_cites = cited = invalid_cites = complete = 0
    possible_cites = 0
    supported_reference = supported_output = correct_supported_output = 0
    state_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in STATE_NAMES}
    for claim_id, target in truth.items():
        guess = guesses.get(claim_id, {"state": "insufficient", "cited_source_ids": []})
        expected_state, actual_state = target["state"], guess["state"]
        supported_reference += expected_state == "supported"
        supported_output += actual_state == "supported"
        correct_supported_output += expected_state == actual_state == "supported"
        correct += expected_state == actual_state
        state_counts[expected_state]["tp" if expected_state == actual_state else "fn"] += 1
        if expected_state != actual_state:
            state_counts[actual_state]["fp"] += 1
        actual = _POLARITIES[actual_state]
        asserted += bool(actual)
        unsupported += bool(actual - _POLARITIES[expected_state])
        citations = set(guess.get("cited_source_ids", []))
        allowed = set(target.get("supporting_source_ids", [])) | set(target.get("refuting_source_ids", []))
        cited += len(citations)
        valid_cites += len(citations & allowed)
        possible_cites += len(allowed)
        if available is not None:
            invalid_cites += len(citations - available)
        if actual:
            groups_by_polarity = {1: target.get("support_sets", []), -1: target.get("refutation_sets", [])}
            complete += all(any(set(group) <= citations for group in groups_by_polarity[polarity])
                            for polarity in actual)
    f1 = []
    for values in state_counts.values():
        denominator = 2 * values["tp"] + values["fp"] + values["fn"]
        if denominator:
            f1.append(2 * values["tp"] / denominator)
    return {
        "state_accuracy": _ratio(correct, len(truth)),
        "state_macro_f1": sum(f1) / len(f1) if f1 else None,
        "state_macro_f1_policy": "average over states present in references or predictions",
        "state_macro_f1_all_four": sum(f1) / 4,
        "state_counts": state_counts,
        "coverage": _ratio(asserted, len(truth)),
        "effective_supported_coverage": _ratio(correct_supported_output, supported_reference),
        "supported_state_error_rate": _ratio(supported_output - correct_supported_output, supported_output),
        "unsupported_claim_rate": _ratio(unsupported, asserted),
        "citation_precision": _ratio(valid_cites, cited),
        "citation_union_recall": _ratio(valid_cites, possible_cites),
        "citation_witness_completeness": _ratio(complete, asserted),
        "withdrawn_or_unknown_citation_rate": (_ratio(invalid_cites, cited) if available is not None else None),
    }


def evaluate_revision(before_reference, after_reference, before_prediction, after_prediction):
    """Evaluate local revision; retained correct predictions cannot hide abstention.

    Supported retention denominator: claims supported before AND after withdrawal.
    The numerator requires a supported prediction in both views.  Change success
    denominator: all claims whose reference state changes, requiring both states.
    """
    old = {row["claim_id"]: row for row in before_reference}
    new = {row["claim_id"]: row for row in after_reference}
    first = {row["claim_id"]: row for row in before_prediction}
    second = {row["claim_id"]: row for row in after_prediction}
    if set(old) != set(new):
        raise ValueError("paired views must use the same claim query IDs")
    kept = retained = changed = revised = unaffected = stable = 0
    for claim_id in old:
        previous, current = old[claim_id]["state"], new[claim_id]["state"]
        pred_previous = first.get(claim_id, {}).get("state", "insufficient")
        pred_current = second.get(claim_id, {}).get("state", "insufficient")
        if previous == current == "supported":
            kept += 1
            retained += pred_previous == pred_current == "supported"
        if previous != current:
            changed += 1
            revised += pred_previous == previous and pred_current == current
        else:
            unaffected += 1
            stable += pred_previous == pred_current == current
    return {"supported_retention": _ratio(retained, kept),
            "required_state_change_success": _ratio(revised, changed),
            "unchanged_state_correctness": _ratio(stable, unaffected)}
