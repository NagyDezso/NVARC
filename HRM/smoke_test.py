"""Cheap end-to-end sanity check.

Loads HRM-Text-1B (no GPU needed if you have ~6 GB free RAM in float32, or
~2.5 GB in bfloat16 on GPU), runs a tiny forward pass, and prints shapes.
Use this before paying for a training run.

    python HRM/smoke_test.py [--cpu]
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from serialize import build_inference_prompt, build_sample, DEFAULT_CONDITION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    print(f"loading {args.model} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True,
    ).to(device).eval()
    print(f"loaded. vocab_size={len(tokenizer)} params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ---- 1. Inference prompt path ------------------------------------------
    demo = [
        {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        {"input": [[1, 1], [0, 0]], "output": [[0, 0], [1, 1]]},
    ]
    test_input = [[1, 0, 1], [0, 1, 0]]
    prompt = build_inference_prompt(demo, test_input, tokenizer)
    print(f"prompt length: {len(prompt['input_ids'])}")

    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([prompt["token_type_ids"]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, token_type_ids=token_type_ids)
    print(f"logits shape: {tuple(out.logits.shape)}")

    eos_id = tokenizer.eos_token_id  # `<|box_end|>` on the HRM tokenizer
    pad_id = tokenizer.pad_token_id  # `<|endoftext|>`
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    gen_text = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
    print("---- generated (untuned base; expect nonsense) ----")
    print(gen_text)
    print("---------------------------------------------------")

    # ---- 2. Training sample path -------------------------------------------
    messages = [
        {"role": "user",      "content": "1 0\n0 1"},
        {"role": "assistant", "content": "0 1\n1 0"},
        {"role": "user",      "content": "1 1\n0 0"},
        {"role": "assistant", "content": "0 0\n1 1"},
    ]
    sample = build_sample(messages, tokenizer, DEFAULT_CONDITION, max_length=4096)
    assert sample is not None
    print(f"train sample: len={len(sample)} "
          f"prompt_tokens={sum(t for t in sample.token_type_ids)} "
          f"target_tokens={sum(1 for l in sample.labels if l != -100)}")

    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([sample.token_type_ids], dtype=torch.long, device=device)
    labels = torch.tensor([sample.labels], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels)
    print(f"training loss (untuned base): {out.loss.item():.4f}")
    print("smoke test OK.")


if __name__ == "__main__":
    main()
