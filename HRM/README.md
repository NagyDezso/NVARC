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
    --max_length 8192

# 4. Sanity-check the model loads + forward + generate works.
uv run --project HRM python HRM/smoke_test.py

# 5. SFT — single GPU. Pick a config (sft_lora / sft_full_small / sft_full).
uv run --project HRM python HRM/train_sft.py --config HRM/configs/sft_full.yaml

# 6. Inference — per-puzzle solver (TTT → turbo-DFS → scoring → selection).
uv run --project HRM python HRM/run_inference.py \
    --checkpoint checkpoints/hrm-arc \
    --tasks data/arc-prize-2025/arc-agi_evaluation_challenges.json \
    --solutions data/arc-prize-2025/arc-agi_evaluation_solutions.json \
    --out submission.json \
    --time-budget-hours 11 \
    --decode-batch 4
```

## Notes

- **Environment**: bitsandbytes (4-/8-bit optimizers) installs by default on
  Linux/macOS; Windows has no wheels, so pick a non-bnb `optim` in the config.
  wandb logging is on by default — `wandb login`, or `WANDB_MODE=offline` /
  `report_to: none` to skip it. Configs use `flex_attention`; HRM's
  `prefix_lm=True` is incompatible with `flash_attention_2` (its 4-D PrefixLM
  mask can't be represented), so use `flex_attention` (default) or `sdpa`.
- **Smoke tests**: `train_sft.py --smoke_test` runs 2 Trainer steps on val;
  `run_inference.py --limit N` solves only the first N puzzles. `--decode-batch`
  sets how many augmentations decode together — lower it to 2 or 1 if HRM's
  large recurrent KV cache OOMs.
- If you ran step 2, point `--tokenizer` / `model.name_or_path` / `--base` at
  `models/HRM-Text-1B-arc` downstream. Datasets can be regenerated from scratch
  with `SDG/scripts/build_datasets.py` instead of step 1.
- HRM is **pre-alignment** — there is no instruction template. We define a minimal
  ARC-specific prompt format in `serialize.py`.
- HRM expects `token_type_ids` to mark the prefix block. Without it, attention is
  fully causal and logits degrade noticeably. `prepare_data.py` and `arc_solver.py`
  both build this mask explicitly.
- The HRM tokenizer condition tags `<|object_ref_start|>` (direct),
  `<|object_ref_end|>` (cot), `<|quad_start|>` (noisy), `<|quad_end|>` (synth)
  are *training-time* conditioning tags. We use `synth,cot` by default
  (`<|quad_end|><|object_ref_end|>`) to nudge structured/explained outputs.
- **Vocabulary cut** (`prepare_tokenizer.py`, Quickstart step 2): HRM ships a
  65,536-token vocab; ARC digit-grids use only a few dozen. Cutting it trims
  ~100M params off the tied embedding, shrinks the logits matmul, and makes the
  cut `embed_tokens`/`lm_head` small enough to LoRA the output head cheaply.
- Hardware: the 24 GB GPU (e.g. RTX 4090) budgets below are for seq 4096. The
  configs now run the context-extension SFT at seq 8192 (~2x), so activation
  memory rises accordingly — expect to lower batch/seq or use a larger GPU.
  No `accelerate launch` needed.
  - **LoRA** (`configs/sft_lora.yaml`) — the easy default. ~8–12 GB total.
  - **Full fine-tune** (`configs/sft_full.yaml`) — also fits, but *only* with
    8-bit AdamW (`optim: adamw_bnb_8bit`, bitsandbytes — Linux/macOS): ~12–16 GB.
    With plain `adamw_torch` the optimizer state alone is 8 GB and a run can
    OOM when activations spike. Use ≥ 40 GB if you want plain fp32 Adam.
