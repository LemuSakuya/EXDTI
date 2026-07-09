#!/usr/bin/env python
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import evaluate
import numpy as np
import datasets
from datasets import load_dataset
datasets.disable_caching()
datasets.disable_progress_bar()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    default_data_collator,
    Seq2SeqTrainingArguments,
)

from dti_utils import (
    build_dti_text,
    build_entity_masks,
    clean_generated_text,
    get_required_field,
    make_readable_prediction_text,
    safe_bleu
)
from models.modeling_t5_dti_mult_task import T5ForConditionalGenerationDTIMultTask


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=False, metadata={"help": "Whether to use a fast tokenizer."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use."}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Whether to use auth token."}
    )


@dataclass
class DataTrainingArguments:
    input_format: str = field(
        default="drug_target",
        metadata={"help": "options: drug_target, drug_smiles_with_caption_target"}
    )
    smiles_caption_model_name: Optional[str] = field(
        default="laituan245/molt5-base-smiles2caption",
        metadata={"help": "SMILES-to-text model used when input_format is drug_smiles_with_caption_target."}
    )
    smiles_caption_cache_path: Optional[str] = field(
        default=None, metadata={"help": "Optional JSON cache to store generated captions between runs."}
    )
    smiles_caption_max_length: int = field(
        default=512, metadata={"help": "Maximum length for SMILES caption generation."}
    )
    smiles_caption_num_beams: int = field(
        default=5, metadata={"help": "Beam size for SMILES caption generation."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The dataset name to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The dataset config name to use."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "An optional test data file."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached evaluation set"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=1024, metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=256, metadata={"help": "The maximum total target sequence length after tokenization."}
    )
    val_max_target_length: Optional[int] = field(
        default=None, metadata={"help": "Validation max target length."}
    )
    pad_to_max_length: bool = field(
        default=False, metadata={"help": "Whether to pad all samples to model maximum sentence length."}
    )
    max_test_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of test examples."}
    )
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams to use for generation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels."}
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    drug_field: str = field(
        default="drug_smiles", metadata={"help": "Drug representation field name."}
    )
    target_field: str = field(
        default="target_sequence", metadata={"help": "Target representation field name."}
    )
    explanation_field: str = field(
        default="explanation", metadata={"help": "Explanation field name."}
    )
    label_field: str = field(
        default="label", metadata={"help": "Binary label field name."}
    )
    drug_prefix: str = field(
        default="DRUG", metadata={"help": "Prefix used before the drug representation."}
    )
    target_prefix: str = field(
        default="TARGET", metadata={"help": "Prefix used before the target representation."}
    )
    space_target_tokens: bool = field(
        default=False, metadata={"help": "Whether to insert spaces between target tokens."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a test file.")
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "Input files should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.report_to = []
    training_args.run_name = None

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        raw_datasets = load_dataset("json", data_files={"test": data_args.test_file}, download_mode="force_redownload")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.max_length = data_args.max_target_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        sep_token="<sep>"
    )

    model = T5ForConditionalGenerationDTIMultTask.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror="tuna"
    )

    enable_smiles_caption = data_args.input_format == "drug_smiles_with_caption_target"
    smiles_caption_tokenizer = None
    smiles_caption_model = None
    smiles_caption_cache: Dict[str, str] = {}
    caption_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if enable_smiles_caption:
        if data_args.smiles_caption_cache_path and os.path.exists(data_args.smiles_caption_cache_path):
            with open(data_args.smiles_caption_cache_path, "r", encoding="utf-8") as cache_file:
                cached_payload = json.load(cache_file)
            if isinstance(cached_payload, dict):
                smiles_caption_cache = {str(k): str(v) for k, v in cached_payload.items()}

        smiles_caption_tokenizer = AutoTokenizer.from_pretrained(
            data_args.smiles_caption_model_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        smiles_caption_model = AutoModelForSeq2SeqLM.from_pretrained(
            data_args.smiles_caption_model_name,
            cache_dir=model_args.cache_dir,
        )
        smiles_caption_model.to(caption_device)
        smiles_caption_model.eval()

    def smiles_to_caption(smiles: str) -> str:
        if not enable_smiles_caption or not smiles:
            return ""
        cached_caption = smiles_caption_cache.get(smiles)
        if cached_caption is not None:
            return cached_caption

        inputs = smiles_caption_tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            max_length=data_args.smiles_caption_max_length,
        ).input_ids.to(caption_device)
        with torch.no_grad():
            outputs = smiles_caption_model.generate(
                inputs,
                num_beams=data_args.smiles_caption_num_beams,
                max_length=data_args.smiles_caption_max_length,
            )
        caption = smiles_caption_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        smiles_caption_cache[smiles] = caption
        return caption

    column_names = raw_datasets["test"].column_names
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(example):
        drug_caption = ""
        if enable_smiles_caption:
            drug_caption = smiles_to_caption(str(get_required_field(example, data_args.drug_field)).strip())

        inputs, drug_text, _ = build_dti_text(example, data_args, drug_caption=drug_caption)
        mask_prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
        if data_args.source_prefix is not None:
            inputs = data_args.source_prefix + inputs
        outputs = str(get_required_field(example, data_args.explanation_field)).strip()
        label = get_required_field(example, data_args.label_field)

        model_inputs, drug_mask, target_mask = build_entity_masks(
            tokenizer,
            inputs,
            drug_text,
            data_args.max_source_length,
            prefix_text=mask_prefix
        )
        target_encodings = tokenizer(
            text_target=[outputs],
            max_length=data_args.max_target_length,
            padding=padding,
            truncation=True,
        )
        label_ids = target_encodings["input_ids"][0]
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            label_ids = [(_label if _label != tokenizer.pad_token_id else -100) for _label in label_ids]

        model_inputs["labels"] = label_ids
        model_inputs["cls_labels"] = float(1) if label else float(0)
        model_inputs["drug_mask"] = drug_mask
        model_inputs["target_mask"] = target_mask
        return model_inputs

    test_dataset = raw_datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache
    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    to_remove_token_list = []
    if tokenizer.bos_token:
        to_remove_token_list.append(tokenizer.bos_token)
    if tokenizer.eos_token:
        to_remove_token_list.append(tokenizer.eos_token)
    if tokenizer.pad_token:
        to_remove_token_list.append(tokenizer.pad_token)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    cls_metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    bleu_metric = evaluate.load("bleu")

    pred_labels = []
    pred_probs = []
    gold_labels = []
    raw_pred_texts = []
    pred_texts = []
    gold_texts = []

    dataloader = DataLoader(
        test_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=data_collator
    )

    for batch in tqdm(dataloader):
        if torch.cuda.is_available():
            batch = {k: v.to("cuda") for k, v in batch.items()}

        batch_size = batch["input_ids"].size(0)
        decoder_start_token_id = model.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=batch["input_ids"].device
        )

        forward_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "drug_mask": batch["drug_mask"],
            "target_mask": batch["target_mask"],
            "decoder_input_ids": decoder_input_ids,
        }

        with torch.no_grad():
            outputs = model(**forward_batch)
            pred_logits = outputs["pred_logits"]
            pred_prob = torch.sigmoid(pred_logits)
            pred = torch.where(pred_prob >= 0.5, 1, 0).cpu().tolist()
            pred_labels += pred
            pred_probs += pred_prob.detach().cpu().tolist()
            gold_labels += [int(x) for x in batch["cls_labels"].cpu().tolist()]

            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                drug_mask=batch["drug_mask"],
                target_mask=batch["target_mask"],
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams,
            )

        batch_pred_texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        cleaned_batch_pred_texts = [clean_generated_text(x, to_remove_token_list) for x in batch_pred_texts]
        raw_pred_texts += cleaned_batch_pred_texts
        pred_texts += [
            make_readable_prediction_text(x, to_remove_token_list, pred_label=label)
            for x, label in zip(batch_pred_texts, pred)
        ]

        labels = batch["labels"].detach().cpu().numpy()
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        batch_gold_texts = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        gold_texts += [clean_generated_text(x, to_remove_token_list) for x in batch_gold_texts]

    cls_result = cls_metric.compute(predictions=pred_labels, references=gold_labels)
    result = {k: round(v, 4) for k, v in cls_result.items()}
    result["bleu"] = round(safe_bleu(bleu_metric, pred_texts, gold_texts), 4)

    pos_pred_texts = [pred_texts[i] for i, value in enumerate(gold_labels) if value == 1]
    pos_gold_texts = [gold_texts[i] for i, value in enumerate(gold_labels) if value == 1]
    neg_pred_texts = [pred_texts[i] for i, value in enumerate(gold_labels) if value == 0]
    neg_gold_texts = [gold_texts[i] for i, value in enumerate(gold_labels) if value == 0]

    result["bleu_pos"] = round(safe_bleu(bleu_metric, pos_pred_texts, pos_gold_texts), 4)
    result["bleu_neg"] = round(safe_bleu(bleu_metric, neg_pred_texts, neg_gold_texts), 4)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    with open(os.path.join(training_args.output_dir, "test_preds.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(pred_texts))
    with open(os.path.join(training_args.output_dir, "test_preds_raw.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(raw_pred_texts))
    with open(os.path.join(training_args.output_dir, "test_pred_labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(int(x)) for x in pred_labels]))
    with open(os.path.join(training_args.output_dir, "test_pred_probs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(float(x)) for x in pred_probs]))
    with open(os.path.join(training_args.output_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        for key, value in sorted(result.items()):
            f.write("%s = %s\n" % (key, value))


def _mp_fn(index):
    main()


if __name__ == "__main__":
    sys.argv.extend(["configs/molt5_dti_finetune_mult_task_inference.json"])
    main()
