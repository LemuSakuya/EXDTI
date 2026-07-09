import csv
import json
import os
import re


def get_required_field(example, field_name):
    if field_name not in example:
        raise KeyError("Missing required field: %s" % field_name)
    return example[field_name]


def normalize_target_text(target_text, space_target_tokens=False):
    target_text = str(target_text).strip()
    if not space_target_tokens:
        return target_text

    compact_target = target_text.replace(" ", "")
    return " ".join(list(compact_target))


def build_dti_text(example, data_args, drug_caption=""):
    drug_rep = str(get_required_field(example, data_args.drug_field)).strip()
    target_rep = normalize_target_text(
        get_required_field(example, data_args.target_field),
        getattr(data_args, "space_target_tokens", False)
    )

    drug_text = "%s %s" % (data_args.drug_prefix, drug_rep)
    if getattr(data_args, "input_format", "drug_target") == "drug_smiles_with_caption_target" and drug_caption:
        drug_text = "%s (%s)" % (drug_text, drug_caption)

    target_text = "%s %s" % (data_args.target_prefix, target_rep)
    inputs = "%s ; %s" % (drug_text, target_text)
    return inputs, drug_text, target_text


def build_entity_masks(tokenizer, inputs, first_segment_text, max_source_length, prefix_text=""):
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding="max_length",
        truncation=True
    )
    first_inputs = tokenizer(
        "%s%s" % (prefix_text, first_segment_text),
        max_length=max_source_length,
        padding="max_length",
        truncation=True
    )
    first_mask = first_inputs["attention_mask"]
    second_mask = [x - y for (x, y) in zip(model_inputs["attention_mask"], first_mask)]
    if sum(first_mask) < max_source_length:
        second_mask[sum(first_mask)] = 0
    return model_inputs, first_mask, second_mask


def clean_generated_text(x_str, to_remove_token_list):
    for to_remove_token in to_remove_token_list:
        x_str = x_str.replace(to_remove_token, "")
    x_str = re.sub(r"<extra_id_\d+>", " ", x_str)
    x_str = re.sub(r"\s+", " ", x_str)
    return x_str.strip()


def make_readable_prediction_text(x_str, to_remove_token_list, pred_label=None):
    x_str = clean_generated_text(x_str, to_remove_token_list)
    x_str = x_str.strip(" ;,:")
    x_str = re.sub(r"\s+([,.;:])", r"\1", x_str)

    malformed = (
        not x_str
        or "dti:" in x_str.lower()
        or x_str.count(";") >= 2
        or len(x_str.split()) < 4
    )
    if malformed and pred_label is not None:
        interaction_text = "positive" if int(pred_label) == 1 else "negative"
        return "Predicted %s DTI interaction, but the generated explanation was malformed." % interaction_text

    if x_str and x_str[-1] not in ".!?":
        x_str += "."
    return x_str


def safe_bleu(exp_metric, predictions, references):
    if not predictions or not references:
        return 0.0
    return exp_metric.compute(predictions=predictions, references=references)["bleu"]


def load_jsonl_records(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl_records(file_path, records):
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_template_csv(file_path):
    templates = {}
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            templates[row[1]] = int(row[0])
    return templates
