"""Convert NVARC text-message datasets to HRM PrefixLM tensors.

Input:  NVARC ``grids_v15/*`` HuggingFace datasets on disk, each row having
        ``puzzle_name: str`` and ``messages: list[{role, content}]``.

Output: HF datasets with columns ``input_ids``, ``token_type_ids``, ``labels``,
        ``length``, ``puzzle_name``.

Usage:
    python HRM/prepare_data.py \\
        --in_dir data/grids_v15 \\
        --out_dir data/hrm_v1 \\
        --tokenizer sapientinc/HRM-Text-1B \\
        --max_length 4096

Budget-limited run (the full ~3.2M set is weeks of 4090 time).
``--max_per_subset`` caps every subset to N shuffled rows, producing a balanced
downsampled mix that keeps every data source represented:

    python HRM/prepare_data.py \\
        --in_dir data/grids_v15 \\
        --out_dir data/hrm_v1_small \\
        --max_per_subset 12000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from serialize import build_sample, DEFAULT_CONDITION


def process_subset(
    in_path: Path,
    out_path: Path,
    tokenizer,
    max_length: int,
    max_per_subset: int | None = None,
    seed: int = 42,
) -> None:
    ds = load_from_disk(str(in_path))
    print(f"[{in_path.name}] loaded {len(ds)} rows")

    if max_per_subset is not None and len(ds) > max_per_subset:
        # Shuffle before capping so the downsample is balanced across puzzles,
        # not just the first N rows on disk.
        ds = ds.shuffle(seed=seed).select(range(max_per_subset))
        print(f"[{in_path.name}] capped to {len(ds)} rows (--max_per_subset)")

    rows = []
    n_dropped = 0
    for row in tqdm(ds, desc=in_path.name):
        sample = build_sample(
            messages=row["messages"],
            tokenizer=tokenizer,
            condition=DEFAULT_CONDITION,
            max_length=max_length,
        )
        if sample is None:
            n_dropped += 1
            continue
        rows.append({
            "input_ids": sample.input_ids,
            "token_type_ids": sample.token_type_ids,
            "labels": sample.labels,
            "length": len(sample),
            "puzzle_name": row["puzzle_name"],
        })

    if n_dropped:
        print(f"[{in_path.name}] dropped {n_dropped} samples exceeding max_length={max_length}")

    out_path.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(out_path))
    print(f"[{in_path.name}] wrote {len(rows)} rows -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory containing NVARC grids_v15/* subset folders")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write HRM-tokenized subsets")
    ap.add_argument("--tokenizer", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--max_per_subset", type=int, default=None,
                    help="Cap each subset to N shuffled rows (budget-limited "
                         "runs). Default: keep every row.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed used when --max_per_subset caps a subset")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Optional explicit subset names; default = every subdir")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"tokenizer vocab size: {len(tokenizer)}")
    print(f"eos_token: {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})")
    print(f"pad_token: {tokenizer.pad_token!r} (id {tokenizer.pad_token_id})")
    # `unk_token` shares id 5 with `pad_token` on the HRM tokenizer, so we
    # can't use the `== unk_token_id` trick to detect missing tokens.
    # Compare the resolved id against the explicit unknown sentinel instead.
    for tok in ["<|im_start|>", "<|im_end|>",
                "<|direct|>", "<|cot|>", "<|noisy|>", "<|synth|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tok not in tokenizer.added_tokens_encoder \
                and tok not in tokenizer.get_vocab():
            raise RuntimeError(f"required special token {tok!r} missing from tokenizer")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    subsets = args.subsets or sorted([p.name for p in in_dir.iterdir() if p.is_dir()])
    for name in subsets:
        process_subset(
            in_dir / name, out_dir / name, tokenizer, args.max_length,
            max_per_subset=args.max_per_subset, seed=args.seed,
        )


if __name__ == "__main__":
    main()
