"""
Pluggable autoregressive teacher: produce ``y_prior`` (HxW uint8) or ``None``.

Implementations may wrap HuggingFace, Unsloth, vLLM, etc.; the **interface** stays
model-agnostic so the thesis and ``pipeline`` do not depend on a vendor name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ArcTestQuery:
    """One test cell: train pairs + single test input (ARC JSON shape)."""

    puzzle_id: str
    train_pairs: List[Dict[str, Any]]
    test_input: List[List[int]]


class AutoregressiveTeacher(Protocol):
    """Callable contract for ``hybrid_arc.pipeline``."""

    def propose(self, query: ArcTestQuery) -> Optional[Any]:
        """Return ``numpy.ndarray`` uint8 HxW (0..9), or ``None`` if unavailable."""


class NullTeacher:
    """Always returns ``None`` (TRM-only / ablation)."""

    def propose(self, query: ArcTestQuery) -> Optional[Any]:
        return None


def numpy_if_valid(arr: Any) -> Optional[Any]:
    """Light validation for uint8 grids."""
    import numpy as np

    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.uint8)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] > 30 or a.shape[1] < 1 or a.shape[1] > 30:
        return None
    if not np.all((a >= 0) & (a <= 9)):
        return None
    return a
