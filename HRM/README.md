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

## Layout

```
HRM/
├── README.md              ← you are here
├── pyproject.toml         ← deps, managed by uv
├── configs/
│   ├── sft_lora.yaml      ← LoRA SFT (default, easiest)
│   ├── sft_full.yaml      ← full fine-tune, 8-bit Adam
│   └── sft_full_small.yaml ← full fine-tune on a capped subset
├── prepare_tokenizer.py   ← (optional) shrink HRM embedding to ARC-only tokens
├── prepare_data.py        ← convert NVARC `grids_v15/*` to HRM PrefixLM samples
├── serialize.py           ← grid <-> text + PrefixLM mask construction
├── train_sft.py           ← single-node multi-GPU SFT loop (HF + accelerate)
├── infer.py               ← inference on ARC-AGI eval set
└── ttft.py                ← test-time fine-tuning per-puzzle (LoRA)
```

## Pipeline

```
NVARC SDG  →  data/grids_v15/{arc2_training, mini, concept, rearc, nvarc_*}
              │
              ▼
       HRM/prepare_data.py        (serialize messages → token_ids + token_type_ids)
              │
              ▼
       data/hrm_v1/{train, val}
              │
              ▼
       HRM/train_sft.py           (HF Trainer, PrefixLM loss, optional LoRA)
              │
              ▼
       checkpoints/hrm-arc/
              │
              ▼
       HRM/infer.py + HRM/ttft.py (per-puzzle LoRA fine-tune, then sample)
              │
              ▼
       submission.json
```

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for environment / dependency management. All commands assume you're at the repo root.

```bash
# 0. Create/sync the venv from HRM/pyproject.toml.
uv sync --project HRM
#    4-/8-bit optimizers (bitsandbytes) install by default on Linux/macOS.
#    On Windows there are no bnb wheels — pick a non-bnb optim in the config.
#    wandb logging is on by default (report_to: wandb in every config).
#    Run `wandb login` first, or `wandb offline` / WANDB_MODE=offline to
#    skip the cloud. Set report_to: none in the config to disable entirely.
#    Attention: configs use flex_attention (built into PyTorch, no extra deps).
#    HRM's prefix_lm=True is incompatible with flash_attention_2 — its 4-D
#    PrefixLM mask cannot be represented by FlashAttention. Use flex_attention
#    (default) or sdpa.

# 1. Fetch NVARC augmented-puzzle datasets from Kaggle (~3.2M puzzles, large).
#    This skips the SDG regeneration step entirely.
bash HRM/download_data.sh                # writes to data/grids_v15/
#    Alternatively, regenerate from scratch:
#    uv run --project HRM python SDG/scripts/build_datasets.py

# 2. Tokenize into HRM PrefixLM tensors (one-off, CPU-only).
#    Budget run: --max_per_subset caps each source for a ~1-2 day 4090 run.
uv run --project HRM python HRM/prepare_data.py \
    --in_dir data/grids_v15 \
    --out_dir data/hrm_v1_small \
    --max_per_subset 12000 \
    --max_length 4096
#    Full run (~3.2M samples, weeks of 4090 time): drop --max_per_subset and
#    use --out_dir data/hrm_v1.

# 3. Sanity-check the model loads + forward + generate works
uv run --project HRM python HRM/smoke_test.py

# 4. Optional Trainer smoke test (2 steps on val)
uv run --project HRM python HRM/train_sft.py --config HRM/configs/sft_lora.yaml --smoke_test

# 5. SFT — single GPU (no accelerate launch needed). Pick a config:
#    sft_lora.yaml        LoRA, fast, easiest. data/hrm_v1*
#    sft_full_small.yaml  full FT on the budget mix (faithful NVARC, ~1-2 days)
#    sft_full.yaml        full FT on the complete 3.2M set (weeks on a 4090)
uv run --project HRM python HRM/train_sft.py --config HRM/configs/sft_full_small.yaml
#    Multi-GPU only: wrap with accelerate launch instead:
#    uv run --project HRM accelerate launch HRM/train_sft.py --config HRM/configs/sft_full_small.yaml

# 6. Eval / submission
uv run --project HRM python HRM/infer.py \
    --checkpoint checkpoints/hrm-arc \
    --tasks external/ARC-AGI-2/data/evaluation \
    --out submission.json
```

## Notes

- HRM is **pre-alignment** — there is no instruction template. We define a minimal
  ARC-specific prompt format in `serialize.py`.
- HRM expects `token_type_ids` to mark the prefix block. Without it, attention is
  fully causal and logits degrade noticeably. `prepare_data.py` and `infer.py`
  both build this mask explicitly.
- The HRM tokenizer condition tags `<|object_ref_start|>` (direct),
  `<|object_ref_end|>` (cot), `<|quad_start|>` (noisy), `<|quad_end|>` (synth)
  are *training-time* conditioning tags. We use `synth,cot` by default
  (`<|quad_end|><|object_ref_end|>`) to nudge structured/explained outputs.
- Hardware: both SFT modes run on a single 24 GB GPU (e.g. RTX 4090) at
  seq 4096 with plain `python` — no `accelerate launch` needed.
  - **LoRA** (`configs/sft_lora.yaml`) — the easy default. ~8–12 GB total.
  - **Full fine-tune** (`configs/sft_full.yaml`) — also fits, but *only* with
    8-bit AdamW (`optim: adamw_bnb_8bit`, bitsandbytes — Linux/macOS): ~12–16 GB.
    With plain `adamw_torch` the optimizer state alone is 8 GB and a run can
    OOM when activations spike. Use ≥ 40 GB if you want plain fp32 Adam.
