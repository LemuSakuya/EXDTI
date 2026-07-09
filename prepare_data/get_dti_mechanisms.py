import argparse
import csv
import json
import os
from collections import Counter

import Levenshtein

from dti_utils import load_jsonl_records, write_jsonl_records
from prepare_data.NLPProcess import NLPProcess


def process_explanation(text):
    return " ".join(str(text).strip().split())


def build_template_table(input_files, explanation_field, label_field):
    positive_explanations = []
    for input_file in input_files:
        for record in load_jsonl_records(input_file):
            if record.get(label_field):
                positive_explanations.append(process_explanation(record[explanation_field]))

    mechanisms, actions, _, _ = NLPProcess(["DRUG", "TARGET"], positive_explanations)
    pairs = []
    for mechanism, action in zip(mechanisms, actions):
        mechanism = mechanism.lower()
        action = action.lower()
        if mechanism == "none" and action == "none":
            continue
        pairs.append("%s_%s" % (mechanism, action))

    pair_counter = Counter(pairs)
    ranked_pairs = pair_counter.most_common()
    templates = {"none_none": 0}
    for idx, (pair_name, _) in enumerate(ranked_pairs, start=1):
        templates[pair_name] = idx
    return templates, pair_counter


def write_template_table(output_file, templates, pair_counter):
    folder = os.path.dirname(output_file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pair_name", "count"])
        for pair_name, idx in sorted(templates.items(), key=lambda x: x[1]):
            writer.writerow([idx, pair_name, pair_counter.get(pair_name, 0)])


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


def update_dataset_labels(input_files, output_files, template_file, explanation_field, label_field, exp_id_field):
    templates = read_template_table(template_file)

    for input_file, output_file in zip(input_files, output_files):
        records = load_jsonl_records(input_file)
        positive_indices = []
        positive_explanations = []

        for idx, record in enumerate(records):
            if record.get(label_field):
                positive_indices.append(idx)
                positive_explanations.append(process_explanation(record[explanation_field]))

        mechanisms, actions, _, _ = NLPProcess(["DRUG", "TARGET"], positive_explanations)
        predicted_pairs = ["%s_%s" % (m.lower(), a.lower()) for m, a in zip(mechanisms, actions)]

        for record in records:
            if not record.get(label_field):
                record[exp_id_field] = 0

        for idx, pair_name in zip(positive_indices, predicted_pairs):
            if pair_name == "none_none":
                records[idx][exp_id_field] = 0
            elif pair_name in templates:
                records[idx][exp_id_field] = templates[pair_name]
            else:
                records[idx][exp_id_field] = find_nearest_template(pair_name, templates)

        write_jsonl_records(output_file, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--template-file", required=True)
    parser.add_argument("--output-files", nargs="+")
    parser.add_argument("--explanation-field", default="explanation")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--exp-id-field", default="exp_id")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    templates, pair_counter = build_template_table(
        args.input_files,
        args.explanation_field,
        args.label_field
    )
    write_template_table(args.template_file, templates, pair_counter)

    if args.build_only:
        return

    if not args.output_files:
        raise ValueError("output-files is required when build-only is not set.")
    if len(args.output_files) != len(args.input_files):
        raise ValueError("output-files must have the same length as input-files.")

    update_dataset_labels(
        args.input_files,
        args.output_files,
        args.template_file,
        args.explanation_field,
        args.label_field,
        args.exp_id_field
    )


if __name__ == "__main__":
    main()
