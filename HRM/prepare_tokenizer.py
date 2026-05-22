"""Shrink HRM-Text-1B's vocabulary to the tokens ARC actually uses.

HRM-Text-1B ships a 65,536-token vocabulary. ARC puzzles serialized as digit
grids only ever touch a few dozen of them: the ten digits, newline, the turn
markers, the condition tags and the tokenizer's bos/eos/pad/unk. The full vocab
spends ~100M parameters on the (tied) embedding matrix and makes the final
logits projection a 65k-wide matmul.

This is the HRM analogue of NVARC's ``ARChitects/cut_tokenizer.ipynb``. It
writes a self-contained cut model directory that downstream code uses with no
changes — point ``--model`` at it for ``train_sft.py`` and ``run_inference.py``:

  * weight tensors whose first dimension is the vocab size have their rows
    index-selected down to the kept token set (catches the tied
    ``embed_tokens.weight`` and, if untied, ``lm_head.weight`` + bias);
  * the tokenizer is renumbered to the kept ids, so it emits 0..N-1 directly
    and stays consistent with the cut model — no remap layer anywhere;
  * ``config.json`` / ``generation_config.json`` get ``vocab_size`` and the
    bos/eos/pad ids rewritten;
  * the custom ``*.py`` modeling code is copied verbatim.

The keep set is fixed: ARC grids serialize to nothing but digits and newlines
(see ``serialize.grid_to_text``), so no dataset scan is needed — the kept ids
are the digits, the newline, and the special/condition tokens.

Usage:
    uv run --project HRM python HRM/prepare_tokenizer.py \\
        --model sapientinc/HRM-Text-1B \\
        --out_dir models/HRM-Text-1B-arc

Surgery is done on the safetensors state dict directly: every tensor whose
leading dim equals the vocab size is selected. This is robust to whatever HRM
names its layers — it does not depend on the custom model class implementing
HuggingFace's embedding-resize hooks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from serialize import (
    COND_COT,
    COND_DIRECT,
    COND_NOISY,
    COND_SYNTH,
    DEFAULT_CONDITION,
    TURN_END_TOKEN,
    TURN_START_TOKEN,
    grid_to_text,
)

# Tokens that must survive the cut even if they never appear in a grid body.
REQUIRED_NAMED_TOKENS = [
    TURN_START_TOKEN,
    TURN_END_TOKEN,
    COND_DIRECT,
    COND_COT,
    COND_NOISY,
    COND_SYNTH,
]


def build_keep_set(tokenizer) -> set[int]:
    """The fixed set of token ids ARC serialization can ever produce.

    ARC grids serialize to nothing but digits and newlines (see
    ``serialize.grid_to_text``), so the keep set is fully determined and needs
    no dataset scan. It is: every special token, the named condition/turn
    tokens, and the digit + newline tokens of the grid body.
    """
    keep: set[int] = set()

    # All registered special tokens (bos/eos/pad/unk, <|im_*|>, tags, ...).
    keep.update(t for t in tokenizer.all_special_ids if t is not None)

    # Tokens depended on by name — fail loudly if the tokenizer lacks one.
    unk = tokenizer.unk_token_id
    for name in REQUIRED_NAMED_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is None or (unk is not None and tid == unk):
            raise SystemExit(f"error: required token {name!r} is not in the tokenizer")
        keep.add(tid)

    # Grid body: digits 0-9 and the row-separating newline. A single fixed
    # two-row grid spanning all ten digits yields every grid-body token id.
    keep.update(tokenizer.encode(
        grid_to_text([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        add_special_tokens=False,
    ))

    # ARC relies on each digit being its own token (1:1 grid mapping).
    for d in "0123456789":
        n = len(tokenizer.encode(d, add_special_tokens=False))
        if n != 1:
            print(f"warning: digit {d!r} tokenizes to {n} tokens, not 1 "
                  f"(grids will not be 1:1 with cells)")

    return keep


def cut_weights(src: str, out_dir: str, kept_ids: list[int],
                old_vocab_size: int) -> None:
    """Index-select every vocab-dimensioned weight tensor and save one shard."""
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        raise SystemExit(f"error: no .safetensors found in {src}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard))

    sel = torch.tensor(kept_ids, dtype=torch.long)
    n_cut = 0
    for key, tensor in list(state.items()):
        if tensor.ndim >= 1 and tensor.shape[0] == old_vocab_size:
            state[key] = tensor.index_select(0, sel).contiguous().clone()
            n_cut += 1
            print(f"  cut {key}: {tuple(tensor.shape)} -> {tuple(state[key].shape)}")
    if n_cut == 0:
        raise SystemExit(
            f"error: no tensor had leading dim == vocab_size ({old_vocab_size}); "
            f"nothing was cut — check the model")

    save_file(state, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    print(f"wrote cut weights ({n_cut} tensor(s) resized)")


def cut_tokenizer_files(out_dir: str, old_to_new: dict[int, int]) -> None:
    """Renumber tokenizer.json and tokenizer_config.json to the kept ids.

    The tokenizer then emits 0..N-1 directly, staying consistent with the cut
    model so no remap layer is needed anywhere downstream.
    """
    # --- tokenizer.json (the fast-tokenizer definition) --------------------
    tj_path = os.path.join(out_dir, "tokenizer.json")
    if not os.path.exists(tj_path):
        raise SystemExit("error: tokenizer.json not found — only fast "
                          "tokenizers can be renumbered")
    with open(tj_path, encoding="utf-8") as f:
        tj = json.load(f)

    # Added (special) tokens carry their own ids.
    added = []
    for tok in tj.get("added_tokens", []):
        if tok["id"] in old_to_new:
            tok = dict(tok)
            tok["id"] = old_to_new[tok["id"]]
            added.append(tok)
    tj["added_tokens"] = sorted(added, key=lambda t: t["id"])

    model = tj["model"]
    if not isinstance(model.get("vocab"), dict):
        raise SystemExit("error: unsupported tokenizer model — expected a BPE "
                         "vocab dict in tokenizer.json")
    model["vocab"] = {tok: old_to_new[oid]
                      for tok, oid in model["vocab"].items()
                      if oid in old_to_new}

    # Keep only merges whose both operands and result still exist.
    kept_strings = set(model["vocab"]) | {t["content"] for t in tj["added_tokens"]}
    if model.get("merges"):
        merges = []
        for m in model["merges"]:
            a, b = (m if isinstance(m, list) else m.split(" "))
            if a in kept_strings and b in kept_strings and (a + b) in kept_strings:
                merges.append(m)
        model["merges"] = merges

    # The post-processor may reference special-token ids; remap them in place.
    def remap_ids(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "ids" and isinstance(val, list):
                    node[key] = [old_to_new.get(i, i) for i in val]
                else:
                    remap_ids(val)
        elif isinstance(node, list):
            for item in node:
                remap_ids(item)

    remap_ids(tj.get("post_processor"))

    with open(tj_path, "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False, indent=2)
    print("renumbered tokenizer.json")

    # --- tokenizer_config.json (added_tokens_decoder is keyed by id) -------
    tc_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path, encoding="utf-8") as f:
            tc = json.load(f)
        decoder = tc.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            tc["added_tokens_decoder"] = {
                str(old_to_new[int(i)]): v
                for i, v in decoder.items()
                if int(i) in old_to_new
            }
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)
        print("renumbered tokenizer_config.json")


def patch_config(path: str, old_to_new: dict[int, int], new_vocab_size: int) -> None:
    """Rewrite vocab_size and the bos/eos/pad token ids of a config file."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        cfg = json.load(f)
    if "vocab_size" in cfg:
        cfg["vocab_size"] = new_vocab_size
    for key in ("bos_token_id", "eos_token_id", "pad_token_id",
                "decoder_start_token_id"):
        old = cfg.get(key)
        if isinstance(old, int):
            if old not in old_to_new:
                raise SystemExit(
                    f"error: {os.path.basename(path)}:{key}={old} is not in the "
                    f"kept token set — it must be force-included")
            cfg[key] = old_to_new[old]
        elif isinstance(old, list):
            cfg[key] = [old_to_new[o] for o in old if o in old_to_new]
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"patched {os.path.basename(path)}")


def verify(out_dir: str, new_vocab_size: int) -> None:
    """Load the cut tokenizer + model and check they agree."""
    from transformers import AutoModelForCausalLM
    print("verifying: loading cut tokenizer + model...")

    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    prompt = (TURN_START_TOKEN + DEFAULT_CONDITION
              + grid_to_text([[1, 2], [3, 4]]) + TURN_END_TOKEN)
    ids = tok.encode(prompt, add_special_tokens=False)
    if max(ids) >= new_vocab_size:
        raise SystemExit(f"error: cut tokenizer emitted id {max(ids)} >= "
                         f"vocab size {new_vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        out_dir, dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=inp, token_type_ids=torch.ones_like(inp))
    if out.logits.shape[-1] != new_vocab_size:
        raise SystemExit(f"error: cut model emits {out.logits.shape[-1]} "
                         f"logits, expected {new_vocab_size}")
    print(f"  ok: tokenizer + model agree on a {new_vocab_size}-token vocab")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B",
                    help="HF hub id or local path of the base HRM model")
    ap.add_argument("--out_dir", required=True,
                    help="destination directory for the cut model")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip the post-cut tokenizer+model agreement check")
    args = ap.parse_args()

    if os.path.isdir(args.model):
        src = args.model
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(args.model)
    print(f"source model: {src}")

    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    old_vocab_size = len(tokenizer)

    keep = build_keep_set(tokenizer)
    kept_ids = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(kept_ids)}
    new_vocab_size = len(kept_ids)
    print(f"keeping {new_vocab_size} / {old_vocab_size} tokens "
          f"({100 * new_vocab_size / old_vocab_size:.2f}%)")

    # Copy everything except the weight shards (they get rewritten).
    os.makedirs(args.out_dir, exist_ok=True)
    skip = {"model.safetensors.index.json"}
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name in skip:
            continue
        s, d = os.path.join(src, name), os.path.join(args.out_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print(f"copied modeling code + tokenizer into {args.out_dir}")

    cut_weights(src, args.out_dir, kept_ids, old_vocab_size)
    cut_tokenizer_files(args.out_dir, old_to_new)
    patch_config(os.path.join(args.out_dir, "config.json"),
                 old_to_new, new_vocab_size)
    patch_config(os.path.join(args.out_dir, "generation_config.json"),
                 old_to_new, new_vocab_size)

    if not args.no_verify:
        verify(args.out_dir, new_vocab_size)

    print(f"""
done. cut model written to: {args.out_dir}

It is a drop-in replacement for the base model — the tokenizer and weights are
both renumbered to the {new_vocab_size}-token ARC vocabulary, so no code needs
to change. Point downstream tooling at it:

  HRM/train_sft.py      set  model.name_or_path: {args.out_dir}  in the config
  HRM/run_inference.py  pass --base {args.out_dir}
""")


if __name__ == "__main__":
    main()
