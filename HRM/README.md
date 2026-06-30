# NVARC × HRM-Text-1B

Adaptation of the [NVARC](../README.md) ARC-AGI-2 solution to use
[sapientinc/HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) as the
reasoning LM, replacing the Qwen3-4B model used in the [ARChitects](../ARChitects)
pipeline.

The [SDG](../SDG) (synthetic data) and [TRM](../TRM) (tiny recursive grid model)
components are unchanged.

## Why HRM-Text-1B

HRM is a hierarchical recurrent transformer: two stacks (H = slow / abstract,
L = fast / concrete) iterate `H_cycles × L_cycles` times over the same input
embeddings, giving effectively unbounded compute depth at 1B parameters. The
recurrent inner loop is conceptually close to TRM's iterative refinement, but
operates over text tokens instead of grid tokens — a natural fit for the
ARChitects "ARC-as-text" formulation.

## What changes vs. ARChitects

| Concern | ARChitects (Qwen3-4B) | HRM-Text-1B port |
|---|---|---|
| Model | Qwen3-4B-Thinking-2507 | sapientinc/HRM-Text-1B |
| Tokenizer | Qwen3 BPE, cut to ARC tokens | HRM tokenizer (65k), optionally cut |
| Training framework | NeMo-RL on Slurm + Megatron TP=8 | HF Transformers `Trainer`, single-GPU by default (`accelerate launch` for multi-GPU) |
| Objective | Causal LM | **PrefixLM** — prompt is bidirectional, response is causal (via `token_type_ids`) |
| Chat template | Qwen3 thinking template | HRM `<|im_start|>{condition}{prompt}<|im_end|>{response}` |
| Vocab cut step | `cut_tokenizer.ipynb` (16 surviving tokens) | `prepare_tokenizer.py` (digits + newline + specials) |
| Test-time fine-tune | LoRA / Unsloth | LoRA via PEFT |

## Inference

`run_inference.py` runs the per-puzzle ARChitects-style solver. Per puzzle it:

1. **TTT** — resets a LoRA adapter and fine-tunes it on 16×8 dihedral+colour
   augmentations of the puzzle's demonstration pairs (`arc_solver.test_time_train`);
2. **turbo-DFS decode** — depth-first search over the token tree, batched across
   augmentations, pruned by cumulative NLL (`max_score = -log(0.2)`)
   (`arc_solver.turbo_dfs`);
3. **augmentation scoring** — re-scores every candidate grid as the *answer*
   under 8 fresh augmentations (`arc_solver.calc_scores`);
4. **selection** — groups identical grids and ranks them with
   `score_full_probmul_3` / `score_kgmon` (`arc_decoder.py`), picking top-2.

PrefixLM `token_type_ids` are threaded through TTT, decode and scoring;
`turbo_dfs` uses `DynamicCache.crop()` to restore the KV cache between sibling
DFS branches.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for environment / dependency management. All commands assume you're at the repo root.

```bash
# 0. Create/sync the venv.
uv sync --project HRM

# 1. Fetch NVARC augmented-puzzle datasets from Kaggle.
bash HRM/download_data.sh                # writes to data/grids_v15/

# 2. (Optional) Cut HRM's 65k vocab to the few-dozen tokens ARC uses.
uv run --project HRM python HRM/prepare_tokenizer.py \
    --model sapientinc/HRM-Text-1B \
    --out_dir models/HRM-Text-1B-arc

# 3. Tokenize into HRM PrefixLM tensors (drop --max_per_subset for the full run).
uv run --project HRM python HRM/prepare_data.py \
    --in_dir data/grids_v15 \
    --out_dir data/hrm_v1_small \
    --tokenizer models/HRM-Text-1B-arc \
    --max_per_subset 12000 \
    --max_length 4096

# 4. Sanity-check the model loads + forward + generate works.
uv run --project HRM python HRM/smoke_test.py

# 5. SFT — single GPU. Pick a config (sft_lora / sft_full_small / sft_full).
uv run --project HRM python HRM/train_sft.py --config HRM/configs/sft_full.yaml

# 6. Inference — per-puzzle solver (TTT → turbo-DFS → scoring → selection).
uv run --project HRM python HRM/run_inference.py \
    --checkpoint checkpoints/hrm-arc-full \
    --tasks data/arc-prize-2025/arc-agi_evaluation_challenges.json \
    --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \
    --out submission.json \
    --time-budget-hours 11 \
    --decode-batch 4 \
    --decode-budget 1800       # cap DECODE per puzzle (0 = global deadline only)
```

## Notes

- **Environment**: bitsandbytes (4-/8-bit optimizers) installs by default on
  Linux/macOS; Windows has no wheels, so pick a non-bnb `optim` in the config.
  wandb logging is on by default — `wandb login`, or `WANDB_MODE=offline` /
  `report_to: none` to skip it. Configs use `flex_attention`; HRM's
  `prefix_lm=True` is incompatible with `flash_attention_2` (its 4-D PrefixLM
  mask can't be represented), so use `flex_attention` (default) or `sdpa`.
- **Smoke tests / debugging**: `train_sft.py --smoke_test` runs 2 Trainer steps
  on val with no checkpointing; `--resume [DIR]` resumes from the latest (or a
  named) checkpoint. `run_inference.py --limit N` solves only the first N puzzles
  and `--keys k1,k2` solves a named subset. `--decode-batch` sets how many
  augmentations decode together — lower it to 2 or 1 if HRM's large recurrent KV
  cache OOMs. `--decode-budget S` caps DECODE-only wall-clock per puzzle in
  seconds (0 = bounded only by `--time-budget-hours`); TTT is never capped by it,
  and `solve_puzzle` keeps whatever candidates it found before the cut, so one
  slow grid can't starve the queue.
- **SFT configs**: all three train at seq 4096, `per_device_train_batch_size 2`,
  on the vocab-cut `models/HRM-Text-1B-arc`. `sft_lora` is a LoRA run (r=32,
  linear schedule, `adamw_torch`); `sft_full_small` and `sft_full` are full
  fine-tunes with cosine annealing and `adamw_bnb_8bit` — `sft_full_small` on a
  budget-capped balanced mix, `sft_full` on the complete dataset.
- **Eval metrics**: SFT eval reports teacher-forced `token_acc` and `grid_exact`
  plus an error breakdown — `token_acc_color` vs `token_acc_struct` (cell colour
  vs grid layout) and `token_acc_q1..q4` (accuracy by quartile of each grid's
  own length, exposing late-grid degradation). These feed the gold prefix at
  every step, so they are optimistic, but they need no decoding.
- **Context length**: `prepare_data.py` defaults `--max_length` to 8192, trimming
  a row's oldest demonstration pairs to fit (the supervised target is never
  dropped; a row is discarded only if its target pair alone overflows). The SFT
  configs and `run_inference.py` (`--max-seq-length`) operate at 4096 — HRM's
  pretrained RoPE window — so the Quickstart tokenizes with `--max_length 4096`
  to match.
- If you ran step 2, point `--tokenizer` / `model.name_or_path` / `--base` at
  `models/HRM-Text-1B-arc` downstream. Datasets can be regenerated from scratch
  with `SDG/scripts/build_datasets.py` instead of step 1.
- HRM is **pre-alignment** — there is no instruction template. We define a minimal
  ARC-specific prompt format in `serialize.py`.
- HRM expects `token_type_ids` to mark the prefix block. Without it, attention is
  fully causal and logits degrade noticeably. `prepare_data.py` and `arc_solver.py`
  both build this mask explicitly.
- The condition tags `<|direct|>`, `<|cot|>`, `<|noisy|>` and `<|synth|>` are
  *training-time* conditioning tags (`serialize.py`). The default
  `DEFAULT_CONDITION` is `<|synth|><|cot|>` (synth + chain-of-thought),
  prepended to every prompt to nudge structured/explained outputs.
- **Vocabulary cut** (`prepare_tokenizer.py`, Quickstart step 2): HRM ships a
  65,536-token vocab; ARC digit-grids use only a few dozen. Cutting it trims
  ~100M params off the tied embedding, shrinks the logits matmul, and makes the
  cut `embed_tokens`/`lm_head` small enough to LoRA the output head cheaply.
