"""ARC <-> HRM-Text-1B serialization.

HRM uses a PrefixLM objective: the prompt block is attended bidirectionally,
the response block is attended causally. The HuggingFace ``hrm_text`` model
class consumes a ``token_type_ids`` tensor where 1 marks prompt tokens
(bidirectional) and 0 marks response tokens (causal). The loss is computed
only on response tokens — prompt token labels are set to -100.

The wire format is (where `<|EOS|>` is the tokenizer's real eos token,
`<|box_end|>` on the HRM tokenizer):

    <|im_start|>{condition}{user_grid_1}<|im_end|>{assistant_grid_1}<|EOS|>
    <|im_start|>{condition}{user_grid_2}<|im_end|>{assistant_grid_2}<|EOS|>
    ...

For ARC, the natural framing is: each puzzle has N demonstration pairs plus 1
test input that must be transformed. We serialize all demonstration
(input, output) pairs followed by the final (test_input, test_output) pair.
Demonstration pairs act as in-context examples; the final assistant block is
what we train to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Condition tags. These are dedicated tokens at ids 1-4 in the HRM tokenizer
# (verified via tokenizer.added_tokens_encoder on sapientinc/HRM-Text-1B).
COND_DIRECT = "<|direct|>"
COND_COT    = "<|cot|>"
COND_NOISY  = "<|noisy|>"
COND_SYNTH  = "<|synth|>"

# Default for ARC: synthetic + chain-of-thought style.
DEFAULT_CONDITION = COND_SYNTH + COND_COT

# Turn boundary inside the prompt block; end-of-response uses the tokenizer's
# real eos_token (`<|box_end|>`, id 11) which we resolve at tokenization time.
TURN_END_TOKEN = "<|im_end|>"
TURN_START_TOKEN = "<|im_start|>"

Grid = list[list[int]]


def grid_to_text(grid: Grid) -> str:
    """ARChitects-compatible serialization: rows of digits joined by '\\n'."""
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def text_to_grid(text: str) -> Grid:
    rows = [r for r in text.strip().split("\n") if r]
    return [[int(c) for c in row] for row in rows]


@dataclass
class Sample:
    """One tokenized PrefixLM training example."""
    input_ids: list[int]
    token_type_ids: list[int]   # 1 = prompt (bidirectional), 0 = response (causal)
    labels: list[int]           # -100 on prompt, copy of input_ids on response

    def __len__(self) -> int:
        return len(self.input_ids)


def encode_sample(
    messages: Sequence[dict],
    tokenizer,
    condition: str = DEFAULT_CONDITION,
) -> Sample:
    """Tokenize an ARChitects-style message list into an HRM PrefixLM sample.

    ``messages`` alternates user/assistant; the *final* assistant message is
    the prediction target. All earlier user/assistant messages become the
    bidirectional prefix (demonstrations + test input).

    Always returns the Sample regardless of length — callers that need a
    length cap should use ``build_sample`` or check ``len(sample)``.
    """
    assert len(messages) >= 2 and len(messages) % 2 == 0, \
        f"need an even count of alternating user/assistant messages, got {len(messages)}"
    assert messages[-1]["role"] == "assistant"
    assert messages[-2]["role"] == "user"

    im_start = tokenizer.convert_tokens_to_ids(TURN_START_TOKEN)
    im_end   = tokenizer.convert_tokens_to_ids(TURN_END_TOKEN)
    eos_id   = tokenizer.eos_token_id
    cond_ids = tokenizer.encode(condition, add_special_tokens=False)

    prompt_ids: list[int] = []
    # Demonstration pairs (all but the last user/assistant) + the final user
    # message form the prefix.
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i]
        prompt_ids.append(im_start)
        prompt_ids.extend(cond_ids)
        prompt_ids.extend(tokenizer.encode(user_msg["content"], add_special_tokens=False))
        prompt_ids.append(im_end)

        # If this isn't the final pair, include the assistant demonstration in
        # the prefix too (it's *in-context*, not a target).
        if i + 1 < len(messages) - 1:
            asst_msg = messages[i + 1]
            prompt_ids.extend(tokenizer.encode(asst_msg["content"], add_special_tokens=False))
            prompt_ids.append(eos_id)

    # Final assistant message = the supervised target.
    target_text = messages[-1]["content"]
    target_ids = tokenizer.encode(target_text, add_special_tokens=False) + [eos_id]

    input_ids = prompt_ids + target_ids
    token_type_ids = [1] * len(prompt_ids) + [0] * len(target_ids)
    labels = [-100] * len(prompt_ids) + list(target_ids)

    return Sample(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels)


def fit_sample(
    messages: Sequence[dict],
    tokenizer,
    condition: str = DEFAULT_CONDITION,
    max_length: int | None = None,
) -> tuple[Sample | None, int]:
    """``encode_sample`` with a length cap, dropping oldest demos to fit.

    ``messages`` alternates user/assistant; the final pair is the supervised
    target and is never dropped. If the encoded sample exceeds ``max_length``,
    demonstration pairs are removed from the front (oldest first), one pair at
    a time, until it fits — the same policy ``build_inference_prompt`` applies
    at inference, so train and inference see prompts shaped the same way.

    Returns ``(sample, n_demos_dropped)``. ``sample`` is None only if the
    final (test-input + target) pair alone still exceeds ``max_length``.
    """
    msgs = list(messages)
    sample = encode_sample(msgs, tokenizer, condition)
    if max_length is None:
        return sample, 0
    n_demos_dropped = 0
    while len(sample) > max_length and len(msgs) > 2:
        msgs = msgs[2:]            # drop the oldest demo (user+assistant) pair
        n_demos_dropped += 1
        sample = encode_sample(msgs, tokenizer, condition)
    if len(sample) > max_length:
        return None, n_demos_dropped
    return sample, n_demos_dropped


def encode_blocks(
    per_msg_ids: Sequence[Sequence[int]],
    cond_ids: Sequence[int],
    im_start: int,
    im_end: int,
    eos_id: int,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Assemble PrefixLM token blocks from already-tokenized message contents.

    ``per_msg_ids`` is the token-id list of each message's ``content``, in
    order, alternating user/assistant with an even count. This is the
    re-tokenization-free core of ``encode_sample``: callers tokenize every
    message exactly once (ideally batched) and reuse the result across trims.

    Returns ``(demo_blocks, final_prefix, final_target)``:
      * ``demo_blocks`` — one fully-formed token block per non-final
        user/assistant pair (the trimmable in-context demonstrations);
      * ``final_prefix`` — the last pair's prompt tokens (bidirectional);
      * ``final_target`` — the last assistant message + eos (the response).
    """
    n = len(per_msg_ids)
    assert n >= 2 and n % 2 == 0, \
        f"need an even count of alternating user/assistant messages, got {n}"
    demo_blocks: list[list[int]] = []
    for i in range(0, n - 2, 2):
        demo_blocks.append([im_start, *cond_ids, *per_msg_ids[i], im_end,
                            *per_msg_ids[i + 1], eos_id])
    final_prefix = [im_start, *cond_ids, *per_msg_ids[n - 2], im_end]
    final_target = [*per_msg_ids[n - 1], eos_id]
    return demo_blocks, final_prefix, final_target


def fit_blocks(
    demo_blocks: Sequence[Sequence[int]],
    final_prefix: Sequence[int],
    final_target: Sequence[int],
    max_length: int | None = None,
) -> tuple[Sample | None, int]:
    """Build a ``Sample`` from ``encode_blocks`` output, dropping oldest demos.

    Equivalent to ``fit_sample`` but operates on pre-tokenized blocks, so
    trimming to ``max_length`` is list slicing rather than re-tokenization.
    Returns ``(sample, n_demos_dropped)``; ``sample`` is None only if the
    final pair alone exceeds ``max_length``.
    """
    tail = len(final_prefix) + len(final_target)
    total = tail + sum(len(b) for b in demo_blocks)
    n_dropped = 0
    if max_length is not None:
        while total > max_length and n_dropped < len(demo_blocks):
            total -= len(demo_blocks[n_dropped])
            n_dropped += 1
        if total > max_length:
            return None, n_dropped
    prompt_ids = [t for b in demo_blocks[n_dropped:] for t in b]
    prompt_ids.extend(final_prefix)
    target_ids = list(final_target)
    return Sample(
        input_ids=prompt_ids + target_ids,
        token_type_ids=[1] * len(prompt_ids) + [0] * len(target_ids),
        labels=[-100] * len(prompt_ids) + target_ids,
    ), n_dropped


def build_sample(
    messages: Sequence[dict],
    tokenizer,
    condition: str = DEFAULT_CONDITION,
    max_length: int | None = None,
) -> Sample | None:
    """``encode_sample`` with a length cap.

    Drops oldest demonstration pairs to fit ``max_length`` (see ``fit_sample``).
    Returns None only if the target pair alone exceeds ``max_length``.
    """
    sample, _ = fit_sample(messages, tokenizer, condition, max_length)
    return sample


def build_inference_prompt(
    demo_pairs: Sequence[dict],
    test_input_grid: Grid,
    tokenizer,
    condition: str = DEFAULT_CONDITION,
    *,
    max_prompt_tokens: int | None = None,
) -> dict:
    """Build a tokenized prompt for ARC inference (no target).

    Returns dict with ``input_ids`` and ``token_type_ids`` (both length L),
    ready to feed into ``model.generate``. The whole returned block is marked
    as prefix (token_type_ids = 1) — generated tokens will get token_type_ids
    = 0 appended by the generation loop.

    If ``max_prompt_tokens`` is given and the full prompt would exceed it,
    demonstration pairs are dropped from the front (oldest first) until it
    fits — matching ``fit_sample``'s training-time policy. The test-input
    block is never dropped. ``num_demos_used`` reports how many demo pairs
    survived.
    """
    im_start = tokenizer.convert_tokens_to_ids(TURN_START_TOKEN)
    im_end   = tokenizer.convert_tokens_to_ids(TURN_END_TOKEN)
    eos_id   = tokenizer.eos_token_id
    cond_ids = tokenizer.encode(condition, add_special_tokens=False)

    def encode_demo(pair) -> list[int]:
        out = [im_start, *cond_ids]
        out.extend(tokenizer.encode(grid_to_text(pair["input"]), add_special_tokens=False))
        out.append(im_end)
        out.extend(tokenizer.encode(grid_to_text(pair["output"]), add_special_tokens=False))
        out.append(eos_id)
        return out

    # Test input + open assistant block — always kept.
    tail: list[int] = [im_start, *cond_ids]
    tail.extend(tokenizer.encode(grid_to_text(test_input_grid), add_special_tokens=False))
    tail.append(im_end)

    demos = [encode_demo(p) for p in demo_pairs]
    if max_prompt_tokens is not None:
        budget = max_prompt_tokens - len(tail)
        # Keep the most recent demos that fit within the budget.
        kept: list[list[int]] = []
        used = 0
        for demo in reversed(demos):
            if used + len(demo) > budget:
                break
            kept.append(demo)
            used += len(demo)
        demos = list(reversed(kept))

    ids: list[int] = [tok for demo in demos for tok in demo]
    ids.extend(tail)

    return {
        "input_ids": ids,
        "token_type_ids": [1] * len(ids),
        "num_demos_used": len(demos),
    }
