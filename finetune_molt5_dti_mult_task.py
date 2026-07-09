#!/usr/bin/env python
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import inspect
import json
import logging
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Dict, Optional

import evaluate
import numpy as np
import datasets
datasets_module = datasets
from datasets import load_dataset
datasets_module.disable_caching()
datasets_module.disable_progress_bar()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch

from dti_utils import (
    build_dti_text,
    build_entity_masks,
    clean_generated_text,
    get_required_field,
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
    from_checkpoint: bool = field(
        default=False, metadata={"help": "Whether load from checkpoint to continue learning"}
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
    task: str = field(
        default="summarization", metadata={"help": "The current DTI task name."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The dataset name to use."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The dataset config name to use."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional evaluation data file."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "An optional test data file."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for preprocessing."}
    )
    preprocess: bool = field(
        default=True, metadata={"help": "Whether to preprocess the dataset."}
    )
    preprocessed_folder: Optional[str] = field(
        default=None, metadata={"help": "Folder to preprocessed data"}
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
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of training examples."}
    )
    max_val_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of validation examples."}
    )
    max_test_samples: Optional[int] = field(
        default=None, metadata={"help": "Truncate the number of test examples."}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "Number of beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels."}
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    ordered_prompt: bool = field(
        default=True, metadata={"help": "Kept for config compatibility."}
    )
    early_stop_epoch_num: int = field(
        default=3, metadata={"help": "Stop training when the metric does not improve."}
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
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        for file_name in [self.train_file, self.validation_file, self.test_file]:
            if file_name is None:
                continue
            extension = file_name.split(".")[-1]
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

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        root_dir, base = os.path.split(training_args.output_dir)
        if not base:
            root_dir, base = os.path.split(root_dir)
        if root_dir and not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if training_args.do_train:
            with open(sys.argv[1], "r", encoding="utf-8") as f:
                with open(os.path.join(root_dir, "trainer_config.json"), "w", encoding="utf-8") as fout:
                    fout.write(f.read())

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                "Output directory (%s) already exists and is not empty. Use --overwrite_output_dir to overcome."
                % training_args.output_dir
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        raw_datasets = load_dataset("json", data_files=data_files, download_mode="force_redownload")

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

    to_remove_token_list = []
    if tokenizer.bos_token:
        to_remove_token_list.append(tokenizer.bos_token)
    if tokenizer.eos_token:
        to_remove_token_list.append(tokenizer.eos_token)
    if tokenizer.pad_token:
        to_remove_token_list.append(tokenizer.pad_token)

    model = T5ForConditionalGenerationDTIMultTask.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror="tuna"
    )

    def _maybe_wrap_forward_for_cache_position(module):
        try:
            signature = inspect.signature(module.forward)
        except (ValueError, TypeError):
            return
        if "cache_position" in signature.parameters:
            return

        original_forward = module.forward

        def forward_with_cache_position(self, *args, cache_position=None, **kwargs):
            return original_forward(*args, **kwargs)

        module.forward = types.MethodType(forward_with_cache_position, module)

    _maybe_wrap_forward_for_cache_position(model)

    enable_smiles_caption = data_args.input_format == "drug_smiles_with_caption_target"
    smiles_caption_tokenizer = None
    smiles_caption_model = None
    smiles_caption_cache: Dict[str, str] = {}
    smiles_caption_cache_dirty = False
    caption_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if enable_smiles_caption:
        if data_args.smiles_caption_cache_path and os.path.exists(data_args.smiles_caption_cache_path):
            try:
                with open(data_args.smiles_caption_cache_path, "r", encoding="utf-8") as cache_file:
                    cached_payload = json.load(cache_file)
                if isinstance(cached_payload, dict):
                    smiles_caption_cache = {str(k): str(v) for k, v in cached_payload.items()}
            except (OSError, json.JSONDecodeError) as cache_err:
                logger.warning("Failed to load SMILES caption cache from %s: %s", data_args.smiles_caption_cache_path, cache_err)

        if data_args.preprocessing_num_workers:
            logger.warning(
                "Disabling multiprocessing for preprocessing to allow SMILES caption generation. Previous preprocessing_num_workers=%s.",
                data_args.preprocessing_num_workers,
            )
            data_args.preprocessing_num_workers = None

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
        nonlocal smiles_caption_cache_dirty
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
        smiles_caption_cache_dirty = True
        return caption

    def save_smiles_caption_cache():
        nonlocal smiles_caption_cache_dirty
        if not enable_smiles_caption or not smiles_caption_cache_dirty or not data_args.smiles_caption_cache_path:
            return
        cache_dir_name = os.path.dirname(data_args.smiles_caption_cache_path)
        if cache_dir_name and not os.path.exists(cache_dir_name):
            os.makedirs(cache_dir_name, exist_ok=True)
        with open(data_args.smiles_caption_cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(smiles_caption_cache, cache_file, ensure_ascii=False, indent=2)
        smiles_caption_cache_dirty = False

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        save_smiles_caption_cache()
        return

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(example):
        if data_args.input_format not in ["drug_target", "drug_smiles_with_caption_target"]:
            raise NotImplementedError

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
            max_length=max_target_length,
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

    datasets_module.enable_progress_bar()
    logger.info("Start Data Preprocessing ...")

    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preprocessing train dataset"
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preprocessing validation dataset"
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = raw_datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Preprocessing test dataset"
        )

    logger.info("End Data Preprocessing ...")
    datasets_module.disable_progress_bar()

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

    exp_metric = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds = [clean_generated_text(x, to_remove_token_list) for x in decoded_preds]
        decoded_labels = [clean_generated_text(x, to_remove_token_list) for x in decoded_labels]
        bleu_score = safe_bleu(exp_metric, decoded_preds, decoded_labels)
        return {"bleu": round(bleu_score, 4)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate and (training_args.do_eval or training_args.do_predict) else None
    )
    if training_args.do_eval:
        trainer.add_callback(EarlyStoppingCallback(data_args.early_stop_epoch_num))

    start_time = time.time()

    if training_args.do_train:
        checkpoint = None
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_result.metrics["total_training_time"] = time.time() - start_time
        trainer.save_model()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w", encoding="utf-8") as writer:
                for key, value in sorted(train_result.metrics.items()):
                    writer.write("%s = %s\n" % (key, value))
                for key, value in sorted(vars(model_args).items()):
                    writer.write("%s = %s\n" % (key, value))
                for key, value in sorted(vars(training_args).items()):
                    writer.write("%s = %s\n" % (key, value))
                for key, value in sorted(vars(data_args).items()):
                    writer.write("%s = %s\n" % (key, value))
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    if training_args.do_predict:
        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        output_test_result_file = os.path.join(training_args.output_dir, "test_results.txt")
        output_test_pred_file = os.path.join(training_args.output_dir, "test_preds.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w", encoding="utf-8") as writer:
                for key, value in sorted(test_results.metrics.items()):
                    writer.write("%s = %s\n" % (key, value))

            test_preds = tokenizer.batch_decode(
                test_results.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            cleaned_preds = [clean_generated_text(x, to_remove_token_list) for x in test_preds]
            with open(output_test_pred_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(cleaned_preds))

    save_smiles_caption_cache()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    sys.argv.extend(["configs/molt5_dti_finetune_mult_task.json"])
    main()
