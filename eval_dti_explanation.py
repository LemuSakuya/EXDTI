import argparse
import csv
import json
from collections import defaultdict

import evaluate
import Levenshtein

from dti_utils import load_jsonl_records
from prepare_data.NLPProcess import NLPProcess


def read_template_table(template_file):
    templates = {}
    with open(template_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            templates[row[1]] = int(row[0])
    return templates


def find_nearest_template(query, templates):
    query_m, query_a = query.split("_", 1)
    template_pairs = [tuple(pair.split("_", 1)) for pair in templates.keys()]
    scores = [Levenshtein.ratio(query_m, pair[0]) if query_a == pair[1] else 0 for pair in template_pairs]
    best_pair = template_pairs[scores.index(max(scores))]
    return templates["%s_%s" % best_pair]


def compute_text_metric(refs, preds):
    results = {}
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    bleu_result = bleu_metric.compute(predictions=preds, references=refs)
    rouge_result = rouge_metric.compute(predictions=preds, references=refs)
    bertscore_result = bertscore_metric.compute(predictions=preds, references=refs, lang="en")

    results["bleu"] = round(bleu_result["bleu"], 4)
    results["rouge1"] = round(rouge_result["rouge1"], 4)
    results["rouge2"] = round(rouge_result["rouge2"], 4)
    results["rougeL"] = round(rouge_result["rougeL"], 4)
    results["rougeLsum"] = round(rouge_result["rougeLsum"], 4)
    results["bertscore_p"] = round(sum(bertscore_result["precision"]) / len(bertscore_result["precision"]), 4)
    results["bertscore_r"] = round(sum(bertscore_result["recall"]) / len(bertscore_result["recall"]), 4)
    results["bertscore_f1"] = round(sum(bertscore_result["f1"]) / len(bertscore_result["f1"]), 4)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--pred-label-file")
    parser.add_argument("--explanation-field", default="explanation")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--exp-id-field", default="exp_id")
    parser.add_argument("--template-file")
    args = parser.parse_args()

    gold_records = load_jsonl_records(args.gold_file)
    with open(args.pred_file, "r", encoding="utf-8") as f:
        pred_texts = [line.strip() for line in f.readlines()]

    if len(gold_records) != len(pred_texts):
        raise ValueError("gold-file and pred-file should contain the same number of records.")

    pred_labels = None
    if args.pred_label_file:
        with open(args.pred_label_file, "r", encoding="utf-8") as f:
            pred_labels = [int(line.strip()) for line in f.readlines() if line.strip()]
        if len(pred_labels) != len(gold_records):
            raise ValueError("pred-label-file and gold-file should contain the same number of records.")

    gold_texts = [str(record[args.explanation_field]).strip() for record in gold_records]
    gold_labels = [int(bool(record.get(args.label_field))) for record in gold_records]

    final_score = defaultdict(dict)
    final_score["overall_score"].update(compute_text_metric(gold_texts, pred_texts))

    pos_preds = [pred_texts[i] for i, label in enumerate(gold_labels) if label == 1]
    pos_golds = [gold_texts[i] for i, label in enumerate(gold_labels) if label == 1]
    neg_preds = [pred_texts[i] for i, label in enumerate(gold_labels) if label == 0]
    neg_golds = [gold_texts[i] for i, label in enumerate(gold_labels) if label == 0]

    if pos_preds:
        final_score["pos_score"].update(compute_text_metric(pos_golds, pos_preds))
    if neg_preds:
        final_score["neg_score"].update(compute_text_metric(neg_golds, neg_preds))

    if pred_labels is not None:
        cls_metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        cls_result = cls_metric.compute(predictions=pred_labels, references=gold_labels)
        for key, value in cls_result.items():
            final_score["binary_cls"][key] = round(value, 4)

    if args.template_file:
        templates = read_template_table(args.template_file)
        active_indices = []
        active_preds = []

        for idx, pred in enumerate(pred_texts):
            if pred_labels is not None and pred_labels[idx] == 0:
                continue
            active_indices.append(idx)
            active_preds.append(pred)

        mechanisms, actions, _, _ = NLPProcess(["DRUG", "TARGET"], active_preds)
        pred_pairs = {}
        for idx, mechanism, action in zip(active_indices, mechanisms, actions):
            pred_pairs[idx] = "%s_%s" % (mechanism.lower(), action.lower())

        exp_label_pred = []
        exp_label_gold = []

        for idx, record in enumerate(gold_records):
            gold_exp_id = int(record.get(args.exp_id_field, 0))
            exp_label_gold.append(gold_exp_id)

            if pred_labels is not None and pred_labels[idx] == 0:
                exp_label_pred.append(0)
                continue

            pred_pair = pred_pairs.get(idx, "none_none")
            if pred_pair == "none_none":
                exp_label_pred.append(0)
            elif pred_pair in templates:
                exp_label_pred.append(templates[pred_pair])
            else:
                exp_label_pred.append(find_nearest_template(pred_pair, templates))

        accuracy = evaluate.load("accuracy")
        recall = evaluate.load("recall")
        precision = evaluate.load("precision")
        f1 = evaluate.load("f1")

        final_score["mult_cls"]["accuracy"] = round(
            accuracy.compute(predictions=exp_label_pred, references=exp_label_gold)["accuracy"], 4
        )
        final_score["mult_cls"]["precision"] = round(
            precision.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")["precision"], 4
        )
        final_score["mult_cls"]["recall"] = round(
            recall.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")["recall"], 4
        )
        final_score["mult_cls"]["f1"] = round(
            f1.compute(predictions=exp_label_pred, references=exp_label_gold, average="macro")["f1"], 4
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_score, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
