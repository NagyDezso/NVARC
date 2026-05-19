"""Test-time fine-tuning (TTFT) for HRM-Text-1B on a single ARC puzzle.

The ARChitects-winning trick: for each test puzzle, take its train pairs and
briefly LoRA-fine-tune the SFT base on permutations of those pairs (with
dihedral + color augmentation), then generate. This adapts the model to the
specific transformation rule the puzzle uses.

Used as a library by infer.py; can also be run standalone for one task to
inspect the per-puzzle loss curve.

Note: this is compute-heavy — N puzzles × K micro-steps × forward+backward at
1B params. Expect to run on GPU with at least 16 GB VRAM.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from serialize import build_sample, DEFAULT_CONDITION, grid_to_text


DIHEDRAL = 8
COLORS = 10


def dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0: return arr
    if tid == 1: return np.rot90(arr, 1)
    if tid == 2: return np.rot90(arr, 2)
    if tid == 3: return np.rot90(arr, 3)
    if tid == 4: return np.fliplr(arr)
    if tid == 5: return np.flipud(arr)
    if tid == 6: return arr.T
    if tid == 7: return np.fliplr(np.rot90(arr, 1))
    raise ValueError(tid)


def color_remap(arr: np.ndarray, mapping: list[int]) -> np.ndarray:
    out = arr.copy()
    for old, new in enumerate(mapping):
        out[arr == old] = new
    return out


def augment_pairs(pairs, rng: random.Random):
    tid = rng.randrange(DIHEDRAL)
    cmap = list(range(COLORS))
    rng.shuffle(cmap)
    out = []
    for p in pairs:
        inp = color_remap(dihedral(np.array(p["input"], dtype=np.uint8), tid), cmap)
        outp = color_remap(dihedral(np.array(p["output"], dtype=np.uint8), tid), cmap)
        out.append({"input": inp.tolist(), "output": outp.tolist()})
    return out


def build_ttft_samples(train_pairs, tokenizer, n_samples: int, max_length: int, seed: int):
    """Build training samples by permuting the train pairs and treating each
    one in turn as the held-out target. Demonstration order is shuffled and
    augmentations (dihedral + color) applied per sample."""
    rng = random.Random(seed)
    samples = []
    n_pairs = len(train_pairs)
    if n_pairs < 2:
        return samples

    for _ in range(n_samples):
        pairs = augment_pairs(train_pairs, rng)
        rng.shuffle(pairs)
        # Take the last pair as the supervised target.
        messages = []
        for p in pairs:
            messages.append({"role": "user",      "content": grid_to_text(p["input"])})
            messages.append({"role": "assistant", "content": grid_to_text(p["output"])})
        s = build_sample(messages, tokenizer, DEFAULT_CONDITION, max_length)
        if s is not None:
            samples.append(s)
    return samples


def ttft_step(model, tokenizer, train_pairs, *,
              n_samples: int = 128, n_steps: int = 32,
              lr: float = 1e-4, max_length: int = 4096, seed: int = 0,
              device: str = "cuda"):
    """Run TTFT in place. Returns the LoRA-augmented model (caller can later
    merge_and_unload() or discard the adapter to reset for the next puzzle).
    """
    from peft import LoraConfig, get_peft_model

    samples = build_ttft_samples(train_pairs, tokenizer, n_samples, max_length, seed)
    if not samples:
        return model

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    pad_id = tokenizer.pad_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    rng = random.Random(seed)
    for step in range(n_steps):
        s = rng.choice(samples)
        input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
        token_type_ids = torch.tensor([s.token_type_ids], dtype=torch.long, device=device)
        labels = torch.tensor([s.labels], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        opt.step()
        opt.zero_grad(set_to_none=True)

    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--task", required=True, help="path to a single ARC task JSON")
    ap.add_argument("--n_samples", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    with open(args.task) as f:
        task = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()

    ttft_step(model, tokenizer, task["train"],
              n_samples=args.n_samples, n_steps=args.n_steps, lr=args.lr,
              device=device)
    print("TTFT complete.")


if __name__ == "__main__":
    main()
