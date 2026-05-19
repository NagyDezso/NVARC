"""ARC-AGI evaluation / submission generation with HRM-Text-1B.

For each puzzle:
  1. Build a prompt with the demonstration pairs + test input.
  2. (Optional) Test-time fine-tune (TTFT): briefly LoRA-tune the base SFT
     checkpoint on permutations of the demonstration pairs so the model adapts
     to the specific puzzle's transformation.
  3. Sample N candidate output grids (with optional dihedral/color augmentation
     at the input + inverse at the output).
  4. Score candidates by log-likelihood, keep top-2 (Kaggle ARC submission
     format expects 2 attempts per test input).

Outputs a Kaggle-format submission.json:
    {puzzle_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}

Usage:
    python HRM/infer.py \\
        --checkpoint checkpoints/hrm-arc \\
        --base sapientinc/HRM-Text-1B \\
        --tasks external/ARC-AGI-2/data/evaluation \\
        --out submission.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from serialize import (
    DEFAULT_CONDITION,
    build_inference_prompt,
    grid_to_text,
    text_to_grid,
)


def load_model(base: str, checkpoint: str | None, dtype: torch.dtype, device: str):
    """Load base HRM model and optionally apply a (LoRA or full) checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, trust_remote_code=True,
    )
    if checkpoint:
        adapter_cfg = os.path.join(checkpoint, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint)
            print(f"loaded LoRA adapter from {checkpoint}")
        else:
            sd = AutoModelForCausalLM.from_pretrained(
                checkpoint, dtype=dtype, trust_remote_code=True,
            ).state_dict()
            model.load_state_dict(sd, strict=False)
            print(f"loaded full-FT weights from {checkpoint}")
    return model.to(device).eval()


@torch.no_grad()
def generate_grid(
    model,
    tokenizer,
    demo_pairs,
    test_input_grid,
    *,
    condition: str = DEFAULT_CONDITION,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """Run a single forward generation pass and decode the predicted grid."""
    device = next(model.parameters()).device
    prompt = build_inference_prompt(demo_pairs, test_input_grid, tokenizer, condition)
    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([prompt["token_type_ids"]], dtype=torch.long, device=device)

    eot_id = tokenizer.eos_token_id  # `<|box_end|>` on the HRM tokenizer

    out = model.generate(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p,
        eos_token_id=eot_id,
        pad_token_id=eot_id,
    )
    gen_ids = out[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    try:
        return text_to_grid(text)
    except Exception:
        return None


def grid_hash(grid):
    return tuple(tuple(int(c) for c in row) for row in grid)


def predict_two_attempts(model, tokenizer, demo_pairs, test_input, n_samples: int = 8):
    """Run 1 greedy + (n_samples-1) sampled generations; majority-vote top-2."""
    candidates = []
    g = generate_grid(model, tokenizer, demo_pairs, test_input, do_sample=False)
    if g is not None:
        candidates.append(g)
    for _ in range(max(0, n_samples - 1)):
        g = generate_grid(
            model, tokenizer, demo_pairs, test_input,
            do_sample=True, temperature=0.7, top_p=0.9,
        )
        if g is not None:
            candidates.append(g)

    if not candidates:
        return [[0]], [[0]]

    counts = Counter(grid_hash(c) for c in candidates)
    top_two = [list(map(list, h)) for h, _ in counts.most_common(2)]
    while len(top_two) < 2:
        top_two.append(top_two[0])
    return top_two[0], top_two[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to SFT output dir (LoRA adapter or full weights)")
    ap.add_argument("--tasks", required=True,
                    help="Directory of ARC-AGI JSON tasks")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = load_model(args.base, args.checkpoint, dtype, args.device)

    submission: dict[str, list[dict]] = {}
    task_files = sorted(glob.glob(os.path.join(args.tasks, "*.json")))
    print(f"found {len(task_files)} tasks under {args.tasks}")

    for path in tqdm(task_files):
        task_id = Path(path).stem
        with open(path) as f:
            task = json.load(f)
        demo_pairs = task["train"]
        attempts_per_test = []
        for test in task["test"]:
            a1, a2 = predict_two_attempts(
                model, tokenizer, demo_pairs, test["input"],
                n_samples=args.n_samples,
            )
            attempts_per_test.append({"attempt_1": a1, "attempt_2": a2})
        submission[task_id] = attempts_per_test

    with open(args.out, "w") as f:
        json.dump(submission, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
