"""Convert NVARC text-message datasets to HRM PrefixLM tensors.

Input:  NVARC ``grids_v15/*`` HuggingFace datasets on disk, each row having
        ``puzzle_name: str`` and ``messages: list[{role, content}]``.

Output: HF datasets with columns ``input_ids``, ``token_type_ids``, ``labels``,
        ``length``, ``puzzle_name``.

Rows longer than ``--max_length`` have their oldest demonstration pairs
trimmed (via ``serialize.fit_blocks``) until they fit; the supervised target
is never dropped. A row is only discarded if its target pair alone overflows.

Usage:
    python HRM/prepare_data.py \\
        --in_dir data/grids_v15 \\
        --out_dir data/hrm_v1 \\
        --tokenizer sapientinc/HRM-Text-1B \\
        --max_length 8192

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

# Each `datasets.map` worker tokenizes single-threaded; the Rust tokenizer's
# own thread pool would oversubscribe cores across `num_proc` processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_from_disk
from serialize import (DEFAULT_CONDITION, TURN_END_TOKEN, TURN_START_TOKEN,
                       encode_blocks, fit_blocks)
from transformers import AutoTokenizer


def _tokenize_batch(batch: dict, tokenizer, max_length: int) -> dict:
    """``datasets.map`` worker: tokenize a batch of rows.

    Runs once per shard in each ``num_proc`` worker process. Every message's
    ``content`` across the whole batch is flattened and tokenized in a single
    batched call -- this amortizes the fast tokenizer's per-call FFI overhead
    and means trimming a row to ``max_length`` (``fit_blocks`` drops oldest
    demo pairs) is list slicing rather than re-tokenization.

    ``n_demos_dropped`` records how many demo pairs were trimmed. A row whose
    target pair alone overflows is marked with ``length = max_length + 1`` and
    empty tensors, and is filtered out downstream.
    """
    im_start = tokenizer.convert_tokens_to_ids(TURN_START_TOKEN)
    im_end = tokenizer.convert_tokens_to_ids(TURN_END_TOKEN)
    eos_id = tokenizer.eos_token_id
    cond_ids = tokenizer.encode(DEFAULT_CONDITION, add_special_tokens=False)

    # Flatten every message content into one list, tokenize in a single Rust
    # call, then slice it back per row. `flat_lens` records each row's message
    # count so we can recover its slice.
    all_messages = batch["messages"]
    flat_texts = [m["content"] for messages in all_messages for m in messages]
    flat_lens = [len(messages) for messages in all_messages]
    flat_ids = tokenizer(flat_texts, add_special_tokens=False)["input_ids"] \
        if flat_texts else []

    out = {"input_ids": [], "token_type_ids": [], "labels": [],
           "length": [], "n_demos_dropped": []}
    pos = 0
    for k in flat_lens:
        per_msg_ids = flat_ids[pos:pos + k]
        pos += k
        demo_blocks, final_prefix, final_target = encode_blocks(
            per_msg_ids, cond_ids, im_start, im_end, eos_id)
        sample, n_demos_dropped = fit_blocks(
            demo_blocks, final_prefix, final_target, max_length)
        out["n_demos_dropped"].append(n_demos_dropped)
        if sample is None:
            out["input_ids"].append([])
            out["token_type_ids"].append([])
            out["labels"].append([])
            out["length"].append(max_length + 1)
            continue
        out["input_ids"].append(sample.input_ids)
        out["token_type_ids"].append(sample.token_type_ids)
        out["labels"].append(sample.labels)
        out["length"].append(len(sample))
    return out


def process_subset(
    in_path: Path,
    out_path: Path,
    tokenizer,
    max_length: int,
    max_per_subset: int | None = None,
    seed: int = 42,
    num_proc: int = 1,
) -> tuple[int, int, int]:
    """Tokenize one subset. Returns (n_kept, n_dropped, n_trimmed)."""
    ds = load_from_disk(str(in_path))
    print(f"[{in_path.name}] loaded {len(ds)} rows")

    if max_per_subset is not None and len(ds) > max_per_subset:
        # Shuffle before capping so the downsample is balanced across puzzles,
        # not just the first N rows on disk.
        ds = ds.shuffle(seed=seed).select(range(max_per_subset))
        print(f"[{in_path.name}] capped to {len(ds)} rows (--max_per_subset)")

    # Tokenize in parallel: `num_proc` worker processes each batch-encode rows;
    # `datasets.map` shards the table across processes and caches the result.
    ds = ds.map(
        _tokenize_batch,
        batched=True,
        num_proc=num_proc if len(ds) >= num_proc else 1,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=["messages"],
        desc=f"tokenize {in_path.name}",
    )

    # Read the bookkeeping columns as plain lists for the keep/trim scan.
    cols = ds.select_columns(["length", "n_demos_dropped"]).to_dict()
    lengths = cols["length"]
    demos_dropped = cols["n_demos_dropped"]
    n_total = len(lengths)
    keep_idx = [i for i, n in enumerate(lengths) if n <= max_length]
    n_dropped = n_total - len(keep_idx)
    n_trimmed = sum(1 for i in keep_idx if demos_dropped[i] > 0)
    total_demos = sum(demos_dropped[i] for i in keep_idx)

    if n_trimmed:
        print(f"[{in_path.name}] trimmed oldest demos on {n_trimmed}/{n_total} "
              f"rows ({100.0 * n_trimmed / n_total:.1f}%) to fit "
              f"max_length={max_length}; {total_demos} demo pairs dropped total")
    if n_dropped:
        print(f"[{in_path.name}] dropped {n_dropped}/{n_total} rows "
              f"({100.0 * n_dropped / n_total:.1f}%) whose target pair alone "
              f"exceeds max_length={max_length}")

    if n_dropped:
        ds = ds.select(keep_idx)
    ds = ds.remove_columns("n_demos_dropped")
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    print(f"[{in_path.name}] wrote {len(ds)} rows -> {out_path}")
    return len(ds), n_dropped, n_trimmed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory containing NVARC grids_v15/* subset folders")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write HRM-tokenized subsets")
    ap.add_argument("--tokenizer", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--max_length", type=int, default=8192,
                    help="Token-length cap. Longer rows have their oldest demo "
                         "pairs trimmed to fit; 8192 keeps ~98.5%% of samples "
                         "while staying inside the trainable context window.")
    ap.add_argument("--max_per_subset", type=int, default=None,
                    help="Cap each subset to N shuffled rows (budget-limited "
                         "runs). Default: keep every row.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed used when --max_per_subset caps a subset")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="Optional explicit subset names; default = every subdir")
    ap.add_argument("--num_proc", type=int, default=os.cpu_count() or 1,
                    help="Worker processes for tokenization (default: all CPUs)")
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
    total_kept = 0
    total_dropped = 0
    total_trimmed = 0
    for name in subsets:
        n_kept, n_dropped, n_trimmed = process_subset(
            in_dir / name, out_dir / name, tokenizer, args.max_length,
            max_per_subset=args.max_per_subset, seed=args.seed,
            num_proc=args.num_proc,
        )
        total_kept += n_kept
        total_dropped += n_dropped
        total_trimmed += n_trimmed

    n_total = total_kept + total_dropped
    pct = 100.0 * total_dropped / n_total if n_total else 0.0
    print(f"[total] wrote {total_kept} rows ({total_trimmed} demo-trimmed), "
          f"dropped {total_dropped}/{n_total} ({pct:.1f}%) whose target pair "
          f"alone exceeds max_length={args.max_length}")


if __name__ == "__main__":
    main()
