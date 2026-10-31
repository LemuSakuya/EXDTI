"""Reproducible action training, paired evidence training, and held-out evaluation.

Run from the project root: python -m exdti_evidence.run --help
"""
import argparse
import hashlib
import json
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, T5Config, T5EncoderModel, T5ForConditionalGeneration, Adafactor

from . import ACTION_LABELS, STATE_LABELS
from .features import FeatureBuilder, pair_text
from .model import ActionPredictor, EvidenceGenerator, paired_loss
from .provenance import intervention_pair, render_claims, release_check

FEATURE_KEYS = ("model_name_or_path", "pair_max_length", "drug_max_length", "protein_max_length",
                "source_max_length", "claim_max_length", "graph_max_length", "max_sources", "max_graph_nodes")


def cast_context(config, device):
    if config.get("bf16") and getattr(device, "type", None) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def enable_checkpointing(module, config):
    if config.get("gradient_checkpointing") and hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()


def check_feature_contract(config, trained_config, keys=FEATURE_KEYS):
    for key in keys:
        if config.get(key) != trained_config.get(key):
            raise ValueError(f"Feature contract mismatch for {key}: training={trained_config.get(key)!r}, evaluation={config.get(key)!r}")


def check_evidence_contract(config, trained_config):
    check_feature_contract(config, trained_config)
    for key, default in (("use_graph", True), ("intervention_mode", "synchronized"), ("condition_on_plan", True)):
        if config.get(key, default) != trained_config.get(key, default):
            raise ValueError(f"Evidence ablation mismatch for {key}; evaluate with the training setting")


def pair_key(record):
    return hashlib.sha256(pair_text(record).encode("utf-8")).hexdigest()


def check_heldout(rows, checkpoint):
    overlap = set(checkpoint.get("training_pair_keys", [])) & {pair_key(row) for row in rows}
    if overlap:
        raise ValueError(f"Evaluation contains {len(overlap)} exact structure/sequence pairs used in training")


def paired_features(record, pair, features, device, config):
    before, after = pair["before"]["evidence"], pair["after"]["evidence"]
    mode = config.get("intervention_mode", "synchronized")
    if mode == "text_only":
        # Deliberately incomplete removal baseline: graph and cached assertions survive.
        observable = deepcopy(before)
        removed = set(after["removed_source_ids"])
        for source in observable["sources"]:
            if source["source_id"] in removed:
                source["text"] = "[text withdrawn]"
        after = observable
    elif mode == "graph_only":
        observable = deepcopy(before)
        observable["graph_assertions"] = deepcopy(after["graph_assertions"])
        observable["artifacts"] = deepcopy(after["artifacts"])
        after = observable
    elif mode != "synchronized":
        raise ValueError("intervention_mode must be synchronized, text_only, or graph_only")
    output = [features(record, view, device) for view in (before, after)]
    if not config.get("use_graph", True):
        for feature in output:
            feature["graph"] = None
            feature["incidence"] = torch.empty((0, len(feature["source_ids"])), device=device)
    return output


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(config):
    random.seed(config.get("seed", 421))
    torch.manual_seed(config.get("seed", 421))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], local_files_only=config.get("local_files_only", False))
    return device, tokenizer, FeatureBuilder(tokenizer, config)


def new_action(config):
    if config.get("tiny_smoke", False):
        cfg = T5Config(vocab_size=config.get("vocab_size", 32128), d_model=32, d_ff=64,
                       num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=16,
                       decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
        return ActionPredictor(T5EncoderModel(cfg))
    return ActionPredictor.from_pretrained(config["model_name_or_path"], local_files_only=config.get("local_files_only", False))


def new_generator(config):
    if config.get("tiny_smoke", False):
        cfg = T5Config(vocab_size=config.get("vocab_size", 32128), d_model=32, d_ff=64,
                       num_layers=1, num_decoder_layers=1, num_heads=2, d_kv=16,
                       decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dropout_rate=0.0)
        model = EvidenceGenerator(T5ForConditionalGeneration(cfg), config.get("freeze_backbone", False))
    else:
        model = EvidenceGenerator.from_pretrained(config["model_name_or_path"],
            freeze_backbone=config.get("freeze_backbone", False), local_files_only=config.get("local_files_only", False))
    model.condition_on_plan = config.get("condition_on_plan", True)
    return model


def optimizer_for(model, config):
    return Adafactor([p for p in model.parameters() if p.requires_grad], lr=config.get("learning_rate", 1e-4),
                     scale_parameter=False, relative_step=False, warmup_init=False)


def output_dir(config):
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoint.pt").exists():
        raise FileExistsError("A checkpoint already exists; select a new output_dir to preserve it")
    write_json(directory / "config.json", config)
    return directory


def load_action(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["action_labels"]) != ACTION_LABELS:
        raise ValueError("This pipeline requires the new three-action checkpoint, not a legacy binary checkpoint")
    model = new_action(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).freeze(), checkpoint


def save_checkpoint(directory, model, config, **metadata):
    torch.save({"state_dict": model.state_dict(), "config": config, "action_labels": ACTION_LABELS,
                "state_labels": STATE_LABELS, **metadata}, directory / "checkpoint.pt")


@torch.no_grad()
def action_eval(model, rows, features, device, config=None):
    config = config or {}
    model.eval()
    confusion = torch.zeros(3, 3, dtype=torch.long)
    calibration = []
    for row in rows:
        label = row.get("action_label")
        if label not in ACTION_LABELS:
            continue
        with torch.no_grad(), cast_context(config, device):
            logits = model(features.pair_tokens(row, device))
        probs = logits.float().softmax(-1)[0]
        pred = probs.argmax(-1).item()
        calibration.append((probs.max().item(), int(pred == ACTION_LABELS.index(label))))
        confusion[ACTION_LABELS.index(label), pred] += 1
    f1 = [2 * confusion[i, i].item() / max(1, confusion[i].sum().item() + confusion[:, i].sum().item()) for i in range(3)]
    ece = 0.0
    for index in range(10):
        members = [(p, y) for p, y in calibration if index / 10 <= p < (index + 1) / 10 or (index == 9 and p == 1)]
        if members:
            ece += abs(sum(p - y for p, y in members)) / max(1, len(calibration))
    return {"n": confusion.sum().item(), "confusion_matrix": confusion.tolist(),
            "macro_f1": sum(f1) / 3, "accuracy": confusion.diag().sum().item() / max(1, confusion.sum().item()),
            "ece_10_equal_width_bins": ece,
            "per_class": {label: {"precision": confusion[i, i].item() / max(1, confusion[:, i].sum().item()),
                                  "recall": confusion[i, i].item() / max(1, confusion[i].sum().item()), "f1": f1[i]}
                          for i, label in enumerate(ACTION_LABELS)}}


def train_action(config):
    device, tokenizer, features = setup(config)
    directory = output_dir(config)
    rows = [r for r in read_rows(config["train_file"]) if r.get("action_label") in ACTION_LABELS]
    if not rows:
        raise ValueError("No valid labelled action training rows")
    model = new_action(config).to(device)
    if config.get("freeze_action_encoder", False):
        model.encoder.requires_grad_(False)
    else:
        enable_checkpointing(model.encoder, config)
    optimizer = optimizer_for(model, config)
    history, start = [], time.time()
    accumulation = config.get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    for step in range(config.get("max_steps", 1000)):
        model.train()
        if config.get("freeze_action_encoder", False):
            model.encoder.eval()
        batch_losses = []
        for _ in range(accumulation):
            row = random.choice(rows)
            inputs = features.pair_tokens(row, device)
            label = torch.tensor([ACTION_LABELS.index(row["action_label"])], device=device)
            with cast_context(config, device):
                loss = F.cross_entropy(model(inputs), label)
            (loss / accumulation).backward()
            batch_losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        history.append({"step": step + 1, "loss": sum(batch_losses) / len(batch_losses)})
        if (step + 1) % config.get("logging_steps", 10) == 0:
            print(json.dumps(history[-1]), flush=True)
    report = {"stage": "action", "run_kind": config.get("run_kind", "research_training"),
              "tiny_smoke": config.get("tiny_smoke", False), "training_rows": len(rows),
              "optimizer_steps": len(history), "elapsed_seconds": time.time() - start, "history": history,
              "train_sha256": fingerprint(config["train_file"])}
    if config.get("dev_file"):
        dev_rows = read_rows(config["dev_file"])
        if config.get("max_eval_records"):
            dev_rows = dev_rows[:config["max_eval_records"]]
        report["dev"] = action_eval(model, dev_rows, features, device, config)
    save_checkpoint(directory, model, config, train_sha256=report["train_sha256"], training_pair_keys=sorted({pair_key(row) for row in rows}))
    write_json(directory / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


def supervision(view, feature, record, tokenizer, config, device):
    gold = {c["claim_id"]: c for c in view["claims"]}
    labels = torch.tensor([STATE_LABELS.index(gold[c]["state"]) for c in feature["claim_ids"]], device=device)
    citations = torch.zeros((len(feature["claim_ids"]), len(feature["source_ids"])), device=device)
    for i, claim_id in enumerate(feature["claim_ids"]):
        supported = set(gold[claim_id]["supporting_source_ids"] + gold[claim_id]["refuting_source_ids"])
        for j, source in enumerate(feature["source_ids"]):
            citations[i, j] = float(source in supported)
    text = render_claims(record, view["claims"])
    target = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].to(device)
    if target.shape[-1] > config.get("target_max_length", 768):
        raise ValueError(f"Reference requires {target.shape[-1]} tokens; raise target_max_length instead of silently dropping claims/citations")
    return labels, citations, target


def choose_removal(record, step):
    """Source to withdraw for one paired view (patent claim 10; [0097], [0118]).

    Use the record's explicit intervention list when present (entry
    ``step mod len(list)``); otherwise select ONE source from the record's
    source list by step index (``step mod len(sources)``).  ``step`` is the
    training step in train-evidence and the record index in evaluate.  All
    copies sharing the selected ``source_unit_id`` are then withdrawn by
    ``build_evidence`` (claim 2).
    """
    # Pair-level partitions are fixed before interventions; views never cross splits.
    sources = record["sources"]
    if not sources:
        return []
    explicit = record.get("interventions", [])
    if explicit:
        entry = explicit[step % len(explicit)]
        return entry["removed_source_ids"] if isinstance(entry, dict) else entry
    return [sources[step % len(sources)]["source_id"]]


def train_evidence(config):
    device, tokenizer, features = setup(config)
    directory = output_dir(config)
    rows = read_rows(config["train_file"])
    if not rows:
        raise ValueError("No evidence training rows")
    action, action_checkpoint = load_action(config["action_checkpoint"], device)
    check_feature_contract(config, action_checkpoint["config"], FEATURE_KEYS[:4])
    if bool(action_checkpoint["config"].get("tiny_smoke", False)) != bool(config.get("tiny_smoke", False)):
        raise ValueError("Cannot mix tiny smoke predictor with the pretrained evidence experiment")
    model = new_generator(config).to(device)
    if not config.get("freeze_backbone", False):
        enable_checkpointing(model.backbone, config)
    optimizer = optimizer_for(model, config)
    history, start = [], time.time()
    accumulation = config.get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    for step in range(config.get("max_steps", 1000)):
        model.train()
        totals = {}
        for microstep in range(accumulation):
            row = random.choice(rows)
            removed = choose_removal(row, step + microstep) if config.get("paired_training", True) else []
            pair = intervention_pair(row, removed)
            # Exactly one prediction call per pair; evidence never enters f(D,T).
            with torch.no_grad(), cast_context(config, device):
                fixed_p = action(features.pair_tokens(row, device)).float().softmax(-1)
            fb, fa = paired_features(row, pair, features, device, config)
            yb, cb, tb = supervision(pair["before"], fb, row, tokenizer, config, device)
            ya, ca, ta = supervision(pair["after"], fa, row, tokenizer, config, device)
            with cast_context(config, device):
                before = model(fb, fixed_p, tb)
                after = model(fa, fixed_p, ta) if config.get("paired_training", True) else before
            loss, pieces = paired_loss(before, after, yb, ya, cb, ca, config.get("loss_weights"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite paired loss")
            (loss / accumulation).backward()
            for name, value in {"total": loss, **pieces}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
        history.append({"step": step + 1, **totals})
        if (step + 1) % config.get("logging_steps", 10) == 0:
            print(json.dumps(history[-1]), flush=True)
    report = {"stage": "evidence", "run_kind": config.get("run_kind", "research_training"),
              "tiny_smoke": config.get("tiny_smoke", False), "training_rows": len(rows),
              "optimizer_steps": len(history), "elapsed_seconds": time.time() - start, "history": history,
              "train_sha256": fingerprint(config["train_file"]), "action_checkpoint_sha256": fingerprint(config["action_checkpoint"]),
              "frozen_predictor_has_grad": any(p.grad is not None for p in action.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save_checkpoint(directory, model, config, action_checkpoint_sha256=report["action_checkpoint_sha256"], training_pair_keys=sorted({pair_key(row) for row in rows}))
    write_json(directory / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


@torch.no_grad()
def evaluate(config):
    device, tokenizer, features = setup(config)
    action, action_checkpoint = load_action(config["action_checkpoint"], device)
    checkpoint = torch.load(config["evidence_checkpoint"], map_location="cpu", weights_only=False)
    check_evidence_contract(config, checkpoint["config"])
    if fingerprint(config["action_checkpoint"]) != checkpoint["action_checkpoint_sha256"]:
        raise ValueError("Evaluation action predictor differs from the one used in paired training")
    model = new_generator(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    records = read_rows(config["test_file"])
    check_heldout(records, checkpoint)
    check_heldout(records, action_checkpoint)
    if config.get("max_eval_records"):
        records = records[:config["max_eval_records"]]
    directory = Path(config["output_dir"]); directory.mkdir(parents=True, exist_ok=True)
    predictions = []
    for step, row in enumerate(records):
        pair = intervention_pair(row, choose_removal(row, step))
        fixed_p = action(features.pair_tokens(row, device)).softmax(-1)
        paired = dict(zip(("before", "after"), paired_features(row, pair, features, device, config)))
        record_output = {"record_id": row["record_id"], "removed_source_ids": choose_removal(row, step),
                         "prediction": dict(zip(ACTION_LABELS, fixed_p[0].cpu().tolist())), "prediction_origin": "frozen_structure_sequence_model"}
        for side in ("before", "after"):
            feature = paired[side]
            generated, out = model.generate(feature, fixed_p, max_new_tokens=config.get("generation_max_tokens", 128), num_beams=1)
            state_ids = out["state_logits"].argmax(-1).tolist()
            cite = out["citation_logits"].sigmoid().cpu().tolist()
            record_output[side] = {"claims": [{"claim_id": cid, "state": STATE_LABELS[state_ids[i]],
                "state_probabilities": out["state_logits"][i].softmax(-1).cpu().tolist(),
                "citation_scores": dict(zip(feature["source_ids"], cite[i])),
                "cited_source_ids": [s for s, score in zip(feature["source_ids"], cite[i]) if score >= config.get("citation_threshold", 0.5)]}
                for i, cid in enumerate(feature["claim_ids"])],
                "raw_generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
                "available_source_ids": pair[side]["evidence"]["source_ids"],
                "oracle_claims_for_scoring_only": pair[side]["claims"],
                "rules_baseline_text": render_claims(row, pair[side]["claims"])}
            if config.get("release_verification", False):
                # Claim 9: raw neural claims above stay untouched for audit.
                record_output[side]["released_claims"] = release_check(
                    row, pair[side]["evidence"], record_output[side]["claims"])
        predictions.append(record_output)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"records": len(predictions), "run_kind": config.get("run_kind", "research_evaluation"),
              "checkpoint_run_kind": checkpoint["config"].get("run_kind", "research_training"),
              "test_sha256": fingerprint(config["test_file"]),
              "intervention_mode": config.get("intervention_mode", "synchronized"),
              "raw_text_semantics_evaluated": False, "release_verification": bool(config.get("release_verification", False)),
              "note": "Structured planner outputs and raw decoder text are saved separately. Do not treat planner accuracy as free-text faithfulness."}
    write_json(directory / "report.json", report)
    print(json.dumps(report), flush=True)


def evaluate_action(config):
    device, _, features = setup(config)
    action, checkpoint = load_action(config["action_checkpoint"], device)
    check_feature_contract(config, checkpoint["config"], FEATURE_KEYS[:4])
    rows = read_rows(config["test_file"])
    check_heldout(rows, checkpoint)
    if config.get("max_eval_records"):
        rows = rows[:config["max_eval_records"]]
    report = {"run_kind": config.get("run_kind", "research_evaluation"),
              "checkpoint_run_kind": checkpoint["config"].get("run_kind", "research_training"),
              "test_sha256": fingerprint(config["test_file"]), **action_eval(action, rows, features, device, config)}
    directory = Path(config["output_dir"]); directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "action_report.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train-action", "train-evidence", "evaluate", "evaluate-action"])
    parser.add_argument("config")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    {"train-action": train_action, "train-evidence": train_evidence, "evaluate": evaluate,
     "evaluate-action": evaluate_action}[args.stage](config)


if __name__ == "__main__":
    main()
