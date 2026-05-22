"""Supervised fine-tuning of HRM-Text-1B on ARC PrefixLM samples.

Single-node multi-GPU SFT using HuggingFace ``Trainer`` + ``accelerate``.
Supports LoRA via PEFT (recommended for 1B at seq 4096) or full fine-tune.

Run:
    accelerate launch HRM/train_sft.py --config HRM/configs/sft_lora.yaml

Smoke test (no accelerate, runs 2 steps on the val set):
    python HRM/train_sft.py --config HRM/configs/sft_lora.yaml --smoke_test
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk, concatenate_datasets
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


def _custom_code_dir(model) -> Path | None:
    """Locate the directory holding HRM's trust_remote_code .py files.

    The HRM architecture is loaded dynamically; its source lives in the HF
    modules cache. We resolve it from the (base) model class file.
    """
    cls = model
    if hasattr(cls, "get_base_model"):       # unwrap PEFT
        cls = cls.get_base_model()
    src = Path(inspect.getfile(cls.__class__))
    return src.parent if src.is_file() else None


class SaveCustomCodeCallback(TrainerCallback):
    """Copy HRM's trust_remote_code .py files into every saved checkpoint.

    HF Trainer checkpoints otherwise omit `modeling_hrm_text.py` /
    `configuration_hrm_text.py`, so `from_pretrained(checkpoint)` cannot
    resolve the custom architecture. Copying them in makes each checkpoint
    self-contained — the config's bare `auto_map` (e.g.
    "modeling_hrm_text.HRMTextForCausalLM") then resolves against the
    checkpoint dir itself.
    """

    def __init__(self, src_dir: Path | None):
        self.src_dir = src_dir

    def _copy_into(self, dest: Path) -> None:
        if self.src_dir is None or not dest.is_dir():
            return
        for py in self.src_dir.glob("*.py"):
            shutil.copy2(py, dest / py.name)

    def on_save(self, args, state, control, **kwargs):
        self._copy_into(Path(args.output_dir) / f"checkpoint-{state.global_step}")


@dataclass
class PrefixLMCollator:
    """Pad input_ids / token_type_ids / labels to the longest sequence in batch.

    HRM's hrm_text model accepts ``input_ids``, ``attention_mask``, and
    ``token_type_ids``. We mask padding with attention_mask=0 and labels=-100.
    """
    pad_token_id: int
    pad_to_multiple_of: int = 64

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, token_type_ids, attention_mask, labels = [], [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            token_type_ids.append(f["token_type_ids"] + [0] * pad)
            attention_mask.append([1] * n + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


def preprocess_logits_for_metrics(logits, labels):
    """Reduce eval logits to argmax token ids before they accumulate.

    Trainer otherwise holds the full [n, seq, vocab] float logits for every
    eval batch — gigabytes at seq 8192. We only need the predicted token id,
    so collapse the vocab axis here and let Trainer accumulate int ids.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Teacher-forced accuracy on the response (target) tokens.

    Not real generative ARC pass@2 — the gold prefix is fed at every step, so
    these numbers are optimistic. But they need no decoding and give a sharp
    per-eval progress curve that plain loss does not.

      * token_acc  — fraction of response tokens whose argmax == gold.
      * grid_exact — fraction of eval rows where *every* response token is
                     correct, i.e. the whole serialized grid (+ EOS) matches.

    The model shifts internally (ForCausalLMLoss), so logits[t] predicts token
    t+1: we compare predictions[:-1] against labels[1:].
    """
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    correct = (preds == labels) & mask

    n_tokens = int(mask.sum())
    token_acc = float(correct.sum()) / n_tokens if n_tokens else 0.0

    row_has_target = mask.any(axis=1)
    row_all_correct = (correct.sum(axis=1) == mask.sum(axis=1)) & row_has_target
    n_rows = int(row_has_target.sum())
    grid_exact = float(row_all_correct.sum()) / n_rows if n_rows else 0.0

    return {"token_acc": token_acc, "grid_exact": grid_exact}


def load_split(paths) -> Any:
    if isinstance(paths, str):
        return load_from_disk(paths)
    parts = [load_from_disk(p) for p in paths]
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def build_model(cfg, tokenizer):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg.model.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=dtype,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        attn_implementation=cfg.model.attn_implementation,
    )
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if cfg.model.lora.enabled:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            target_modules=list(cfg.model.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run 2 training steps on the val set, no checkpointing")
    ap.add_argument("--resume", nargs="?", const=True, default=False,
                    help="Resume training. Pass a checkpoint dir, or no value "
                         "to auto-pick the latest in training.output_dir.")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(cfg.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    print(f"pad_token_id = {pad_id}, vocab_size = {len(tokenizer)}")

    model = build_model(cfg, tokenizer)

    if args.smoke_test:
        train_ds = load_split(cfg.data.val_dataset_path)
        eval_ds = train_ds.select(range(min(8, len(train_ds))))
        training_args_kwargs = dict(
            output_dir="/tmp/hrm-smoke",
            num_train_epochs=1,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            bf16=True,
            report_to=[],
        )
    else:
        train_ds = load_split(list(cfg.data.train_dataset_path))
        eval_ds = load_split(cfg.data.val_dataset_path)
        training_args_kwargs = dict(
            output_dir=cfg.training.output_dir,
            seed=cfg.training.seed,
            num_train_epochs=cfg.training.num_train_epochs,
            max_steps=cfg.training.max_steps,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            max_grad_norm=cfg.training.max_grad_norm,
            logging_steps=cfg.training.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            # Offload eval predictions to CPU every few batches so the
            # accumulated argmax tensors don't pin GPU memory during eval.
            eval_accumulation_steps=8,
            save_strategy="steps",
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            bf16=cfg.training.bf16,
            optim=cfg.training.optim,
            report_to=[cfg.training.report_to] if cfg.training.report_to != "none" else [],
            run_name=cfg.training.run_name,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )

    print(f"train rows: {len(train_ds)}, eval rows: {len(eval_ds)}")

    collator = PrefixLMCollator(pad_token_id=pad_id)
    code_dir = _custom_code_dir(model)
    callbacks = [SaveCustomCodeCallback(code_dir)] if not args.smoke_test else []
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # --resume: True -> let Trainer auto-find the latest checkpoint in
    # output_dir; a string -> resume from that exact checkpoint dir.
    resume = args.resume if not args.smoke_test else False
    trainer.train(resume_from_checkpoint=resume)

    if not args.smoke_test:
        trainer.save_model(cfg.training.output_dir)
        tokenizer.save_pretrained(cfg.training.output_dir)
        # Final save isn't a checkpoint-N dir, so copy the custom code here too.
        SaveCustomCodeCallback(code_dir)._copy_into(Path(cfg.training.output_dir))


if __name__ == "__main__":
    main()
