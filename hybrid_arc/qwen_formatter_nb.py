"""
Qwen prompt / decode helpers aligned with ``arc2-qwen3-unsloth-flash-lora-batch4-queue.ipynb`` (``arc_loader.QwenFormatter``).

The checked-in notebook export encodes the ChatML end marker with a ``redacted_`` placeholder segment;
``_chat_im_end()`` strips the placeholder to the usual Qwen chat end-of-message token. Override with env ``QWEN_CHAT_IM_END`` if your tokenizer differs.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from transformers import PreTrainedTokenizerBase


def _chat_im_end() -> str:
    # Notebook export uses middle segment ``redacted_im_end``; Qwen ChatML end is ``im_end``.
    censored = "<|{}|>".format("redacted_im_end")
    return os.environ.get("QWEN_CHAT_IM_END", censored.replace("redacted_", ""))


def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()


def is_valid_solution(guess) -> bool:
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)


class QwenFormatter:
    """Same structure as notebook ``arc_loader.QwenFormatter``."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, im_end: str | None = None) -> None:
        self.tokenizer = tokenizer
        self._im_end = im_end or _chat_im_end()

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + f"{self._im_end}<|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + self._im_end

    def fmt_train(self, train, last_is_challenge: bool = False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += (
                f"<|im_start|>user\n{grid_input}{self._im_end}<|im_start|>assistant\n{grid_output}{self._im_end}"
            )
        if test is not None:
            text += self.fmt_query([test]) + (self.fmt_reply([test["output"]]) if "output" in test else "")
        return text

    def max_new_tokens(self) -> int:
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows: int = 30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except Exception:
            pass
        return None


def prompt_for_test_puzzle(train: List[dict], test_item: dict, formatter: QwenFormatter) -> str:
    """Notebook-style: all train pairs + challenge test input (no gold output)."""
    train_block = formatter.fmt_train(train, last_is_challenge=False)
    return train_block + formatter.fmt_query([test_item])
