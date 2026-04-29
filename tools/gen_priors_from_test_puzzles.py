#!/usr/bin/env python3
"""
Build ``priors.json`` from TRM eval ``test_puzzles.json`` using the **same stack** as
``arc2-qwen3-unsloth-flash-lora-batch4-queue.ipynb`` (Unsloth ``FastLanguageModel``, notebook
``QwenFormatter`` / token decode).

Dataset: https://www.kaggle.com/datasets/cpmpml/arc-prize-trm-evaluation-data
→ use ``test_puzzles.json`` inside the bundle.

Example (merged / SFT weights only, no runtime LoRA)::

  uv run python tools/gen_priors_from_test_puzzles.py \\
    --test-puzzles TRM/arc-prize-trm-evaluation-data/test_puzzles.json \\
    --out TRM/priors.json \\
    --model /path/to/qwen3_4b_grids15_sft139/transformers/bfloat16/1 \\
    --no-lora \\
    --max-puzzles 10

Example (same as notebook: base + PEFT skeleton; optional pickled LoRA state)::

  uv run python tools/gen_priors_from_test_puzzles.py \\
    --test-puzzles TRM/arc-prize-trm-evaluation-data/test_puzzles.json \\
    --out TRM/priors.json \\
    --model /path/to/base/transformers/bfloat16/1 \\
    --lora-state /path/to/default_weights.pkl \\
    --max-puzzles 10

Verify-then-seed (also asks Qwen to reproduce each train output via leave-one-out, and
emits the per-puzzle confidence the consumer reads as ``pass_rate``)::

  uv run python tools/gen_priors_from_test_puzzles.py \\
    --test-puzzles TRM/arc-prize-trm-evaluation-data/test_puzzles.json \\
    --out TRM/priors.json --verify \\
    --model /path/to/qwen3_4b_grids15_sft139/transformers/bfloat16/1 \\
    --no-lora --max-puzzles 10

Without ``--verify`` the output is the legacy raw-grid schema (consumer treats those
as ``pass_rate=1.0``). With ``--verify`` each test entry becomes
``{"grid": [...], "pass_rate": float, "train_matches": k, "train_total": n}``.
Cost is ``(1 + n_train_pairs)x`` more LLM calls per puzzle.

Requires: ``unsloth``, ``torch``, ``transformers``, ``peft``, ``tqdm`` (Linux + GPU typical).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _notebook_peft_params() -> dict:
    """``arc_solver.py`` / notebook ``get_peft_model`` kwargs."""
    return dict(
        r=256,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="priors.json from test_puzzles.json (NVARC Qwen notebook stack)")
    ap.add_argument("--test-puzzles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Same as notebook FastLanguageModel.from_pretrained(model_name=...)",
    )
    ap.add_argument("--max-seq-length", type=int, default=8192, help="Notebook uses 8192")
    ap.add_argument("--load-in-4bit", action="store_true", help="Notebook default is False")
    ap.add_argument("--local-files-only", action="store_true", help="Notebook uses True on Kaggle")
    ap.add_argument("--no-lora", action="store_true", help="Skip get_peft_model (merged full weights)")
    ap.add_argument(
        "--lora-state",
        type=Path,
        default=None,
        help="Optional pickle of PEFT state dict (torch.load); applied via set_peft_model_state_dict like notebook",
    )
    ap.add_argument("--max-puzzles", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = use QwenFormatter.max_new_tokens()")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Leave-one-out: query Qwen on each train input (using the other train pairs as few-shot) "
        "and compute pass_rate = #matches / #train_pairs. Emits the new {grid, pass_rate} schema. "
        "Single-train-pair puzzles default to pass_rate=1.0 (cannot leave-one-out).",
    )
    args = ap.parse_args()
    if args.lora_state is not None and args.no_lora:
        ap.error("--lora-state requires LoRA (--no-lora must not be set)")

    repo_root = Path(__file__).resolve().parent.parent
    rp = str(repo_root)
    if rp not in sys.path:
        sys.path.insert(0, rp)

    import torch
    from tqdm import tqdm
    from unsloth import FastLanguageModel

    from hybrid_arc.qwen_formatter_nb import QwenFormatter, prompt_for_test_puzzle

    raw = json.loads(Path(args.test_puzzles).read_text(encoding="utf-8"))
    ids = sorted(raw.keys())
    if args.max_puzzles > 0:
        ids = ids[: args.max_puzzles]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        full_finetuning=False,
        load_in_4bit=bool(args.load_in_4bit),
        local_files_only=bool(args.local_files_only),
        use_gradient_checkpointing=False,
        max_seq_length=int(args.max_seq_length),
    )

    if not args.no_lora:
        model = FastLanguageModel.get_peft_model(model, **_notebook_peft_params())
        for name, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        if args.lora_state is not None:
            from peft import set_peft_model_state_dict

            blob = torch.load(args.lora_state, map_location="cpu")
            if not isinstance(blob, dict):
                raise TypeError("--lora-state must be a pickled state_dict (dict of tensors)")
            set_peft_model_state_dict(model, blob, adapter_name="default")

    model = FastLanguageModel.for_inference(model)

    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new = int(args.max_new_tokens) if args.max_new_tokens > 0 else formatter.max_new_tokens()

    def _query(train_pairs: List[Any], query_input: Dict[str, Any]) -> Any:
        """Run Qwen on (few-shot ``train_pairs``, ``query_input``); return uint8 grid or ``None``."""
        prompt = prompt_for_test_puzzle(train_pairs, query_input, formatter)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return formatter.convert_tokens_to_array(gen.detach().cpu().tolist())

    def _grids_equal(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        try:
            import numpy as _np  # noqa: WPS433

            aa = _np.asarray(a)
            bb = _np.asarray(b)
            return aa.shape == bb.shape and bool((aa == bb).all())
        except Exception:
            return False

    priors: Dict[str, List[Any]] = {}

    for pid in tqdm(ids):
        doc = raw[pid]
        train = doc["train"]
        tests = doc["test"]

        # Leave-one-out verification: ask Qwen to reproduce each train output using the
        # remaining train pairs as few-shot. Pass rate is the fraction it nails.
        train_matches = 0
        train_total = 0
        if args.verify and len(train) >= 2:
            train_total = len(train)
            for i, gold in enumerate(train):
                heldout = [p for j, p in enumerate(train) if j != i]
                query_in = {"input": gold["input"]}
                pred = _query(heldout, query_in)
                if _grids_equal(pred, gold["output"]):
                    train_matches += 1
            pass_rate = train_matches / train_total
        else:
            # Single train pair (or --verify off): default to full confidence so γ is
            # unchanged from the legacy path. Consumer ignores pass_rate when reading the
            # legacy raw-grid form.
            pass_rate = 1.0

        entries: List[Any] = []
        for t in tests:
            test_in = {k: v for k, v in t.items() if k != "output"}
            grid = _query(train, test_in)
            grid_l = grid.tolist() if grid is not None else None
            if args.verify:
                if grid_l is None:
                    entries.append(None)
                else:
                    entries.append(
                        {
                            "grid": grid_l,
                            "pass_rate": pass_rate,
                            "train_matches": train_matches,
                            "train_total": train_total,
                        }
                    )
            else:
                entries.append(grid_l)
        priors[pid] = entries

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(priors), encoding="utf-8")
    print(f"Wrote {args.out} ({len(priors)} puzzles, schema={'verify' if args.verify else 'legacy'})")


if __name__ == "__main__":
    main()
