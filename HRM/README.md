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
| Training framework | NeMo-RL on Slurm + Megatron TP=8 | HF Transformers + `accelerate` / `torchrun` |
| Objective | Causal LM | **PrefixLM** — prompt is bidirectional, response is causal (via `token_type_ids`) |
| Chat template | Qwen3 thinking template | HRM `<|im_start|>{condition}{prompt}<|im_end|>{response}` |
| Vocab cut step | `cut_tokenizer.ipynb` (16 surviving tokens) | `prepare_tokenizer.py` (digits + newline + specials) |
| Test-time fine-tune | LoRA / Unsloth | LoRA via PEFT |

## Layout

```
HRM/
├── README.md              ← you are here
├── requirements.txt       ← pinned deps
├── configs/
│   └── sft.yaml           ← training hyperparameters
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
# 0. Create/sync the venv from HRM/pyproject.toml
uv sync --project HRM
#    To include 4-/8-bit optimizers:    uv sync --project HRM --extra quant
#    To include wandb logging:          uv sync --project HRM --extra wandb
#    To include flash-attn:             uv sync --project HRM --extra flash

# 1. Fetch NVARC augmented-puzzle datasets from Kaggle (~3.2M puzzles, large).
#    This skips the SDG regeneration step entirely.
bash HRM/download_data.sh                # writes to data/grids_v15/
#    Alternatively, regenerate from scratch:
#    uv run --project HRM python SDG/scripts/build_datasets.py

# 2. Tokenize into HRM PrefixLM tensors (one-off, CPU-only)
uv run --project HRM python HRM/prepare_data.py \
    --in_dir data/grids_v15 \
    --out_dir data/hrm_v1 \
    --max_length 4096

# 3. Sanity-check the model loads + forward + generate works
uv run --project HRM python HRM/smoke_test.py

# 4. Optional Trainer smoke test (2 steps on val)
uv run --project HRM python HRM/train_sft.py --config HRM/configs/sft.yaml --smoke_test

# 5. SFT (multi-GPU)
uv run --project HRM accelerate launch HRM/train_sft.py --config HRM/configs/sft.yaml

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
- Local 4 GB GPU is sufficient only for tokenizer prep and inference smoke
  tests. Full SFT needs ≥ 1× H100 (or 2–4× A100/RTX 6000); a single H100 fits
  the 1B model in bf16 with LoRA at seq 4096.
