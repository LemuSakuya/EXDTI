#!/usr/bin/env python
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import time
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import datasets
from datasets import load_dataset
datasets.disable_caching()
datasets.disable_progress_bar()

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

from dti_utils import build_dti_text, clean_generated_text, get_required_field


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
        default="drug_target", metadata={"help": "options: drug_target"}
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
        default="label", metadata={"help": "Optional binary label field name."}
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
        raw_datasets = load_dataset("json", data_files=data_files, keep_in_memory=True)

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

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror="tuna"
    )

    to_remove_token_list = []
    if tokenizer.bos_token:
        to_remove_token_list.append(tokenizer.bos_token)
    if tokenizer.eos_token:
        to_remove_token_list.append(tokenizer.eos_token)
    if tokenizer.pad_token:
        to_remove_token_list.append(tokenizer.pad_token)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(example):
        if data_args.input_format != "drug_target":
            raise NotImplementedError

        inputs, _, _ = build_dti_text(example, data_args)
        if data_args.source_prefix is not None:
            inputs = data_args.source_prefix + inputs
        outputs = str(get_required_field(example, data_args.explanation_field)).strip()

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
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
        return model_inputs

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
            load_from_cache_file=not data_args.overwrite_cache
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
            load_from_cache_file=not data_args.overwrite_cache
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
            load_from_cache_file=not data_args.overwrite_cache
        )

    logger.info("End Data Preprocessing ...")

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
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [clean_generated_text(x, to_remove_token_list) for x in decoded_preds]
        decoded_labels = [clean_generated_text(x, to_remove_token_list) for x in decoded_labels]
        bleu_result = exp_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": round(bleu_result["bleu"], 4)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )
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
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            cleaned_preds = [clean_generated_text(x, to_remove_token_list) for x in test_preds]
            with open(output_test_pred_file, "w", encoding="utf-8") as writer:
                writer.write("\n".join(cleaned_preds))


def _mp_fn(index):
    main()


if __name__ == "__main__":
    sys.argv.extend(["configs/molt5_dti_finetune_joint_train.json"])
    main()
