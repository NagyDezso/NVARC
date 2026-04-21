"""
ARC grid ↔ TRM flat token layout (must match ``TRM/dataset/build_arc_dataset.py``).

TRM ``dataset`` imports are **lazy** so callers can prepend ``TRM/`` to ``sys.path`` first.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch

# Duplicated constant to avoid importing ``dataset`` at module import time.
ARCMaxGridSize = 30


def flat_tokens_to_padded_grid(flat: np.ndarray) -> np.ndarray:
    """``flat`` length 900, values PAD=0, EOS=1, colors 2..11 as stored."""
    g = flat.reshape(ARCMaxGridSize, ARCMaxGridSize)
    return g


def crop_grid_from_padded(padded: np.ndarray) -> np.ndarray:
    """Maximum axis-aligned rectangle avoiding EOS (same spirit as evaluator ``_crop``)."""
    grid = padded.reshape(ARCMaxGridSize, ARCMaxGridSize)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) or (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return (grid[: max_size[0], : max_size[1]] - 2).astype(np.uint8)


def input_flat_to_inp_grid(input_flat: np.ndarray) -> np.ndarray:
    """Decode TRM ``inputs`` row to uint8 input grid (0..9)."""
    padded = flat_tokens_to_padded_grid(np.asarray(input_flat, dtype=np.int64))
    return crop_grid_from_padded(padded)


def label_flat_from_grids(inp_grid: np.ndarray, out_grid: np.ndarray, do_translation: bool = False) -> np.ndarray:
    """Return label flat sequence (length 900) for ``out_grid`` paired with ``inp_grid``."""
    from dataset.build_arc_dataset import np_grid_to_seq_translational_augment

    _, out_flat = np_grid_to_seq_translational_augment(inp_grid, out_grid, do_translation=do_translation)
    return np.asarray(out_flat, dtype=np.int64)


def y_prior_to_label_flat(
    input_flat: np.ndarray,
    y_prior: np.ndarray,
    *,
    do_translation: bool = False,
) -> np.ndarray:
    """
    Build TRM label token ids for a proposal grid ``y_prior`` (HxW, 0..9) consistent
    with the padding/EOS layout implied by this row's ``input_flat``.
    """
    from dataset.build_arc_dataset import arc_grid_to_np

    inp = input_flat_to_inp_grid(input_flat)
    out = arc_grid_to_np(y_prior.tolist() if hasattr(y_prior, "tolist") else y_prior)
    return label_flat_from_grids(inp, out, do_translation=do_translation)


def row_augment_from_identifier_name(name: str) -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """``inverse_aug`` from TRM build_arc_dataset: maps augmented coords → canonical."""
    from dataset.build_arc_dataset import inverse_aug

    return inverse_aug(name)


def canonical_grid_to_row_space(grid: np.ndarray, augmented_identifier_name: str) -> np.ndarray:
    """
    Map a **canonical** (base puzzle) ``grid`` into the augmented frame used by TRM
    row ``augmented_identifier_name`` (must match ``identifiers.json`` entries).

    Mirrors ``dataset.build_arc_dataset.aug`` forward map:
    ``dihedral_transform(mapping[grid], trans_id)``.
    """
    from dataset.build_arc_dataset import PuzzleIdSeparator
    from dataset.common import dihedral_transform

    g = np.asarray(grid, dtype=np.uint8)
    if PuzzleIdSeparator not in augmented_identifier_name:
        return g
    parts = augmented_identifier_name.split(PuzzleIdSeparator)
    trans_id = int(parts[-2][1:])
    perm = parts[-1]
    mapping = np.array([int(c) for c in perm], dtype=np.uint8)
    if mapping.size != 10:
        raise ValueError(f"Expected 10 color-map digits in aug name, got {mapping.size}: {augmented_identifier_name!r}")
    return dihedral_transform(mapping[g], trans_id).astype(np.uint8)


def y_prior_tokens_torch(
    input_flat: np.ndarray,
    y_prior: np.ndarray,
    *,
    do_translation: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """``[1, seq_len]`` int64 token ids for embed_tokens."""
    flat = y_prior_to_label_flat(input_flat, y_prior, do_translation=do_translation)
    t = torch.from_numpy(flat).long().unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def batch_y_prior_tokens(
    inputs: torch.Tensor,
    y_priors: list[Optional[np.ndarray]],
    *,
    do_translation: bool = False,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Stack per-row ``y_prior`` into ``[B, seq_len]``. Rows with ``None`` use all -1
    (caller should disable seeding for that batch or filter).
    """
    if all(y is None for y in y_priors):
        return None
    rows = []
    for i, y in enumerate(y_priors):
        if y is None:
            rows.append(torch.full((inputs.shape[1],), -1, dtype=torch.long, device=device))
        else:
            flat = y_prior_to_label_flat(inputs[i].detach().cpu().numpy(), y, do_translation=do_translation)
            rows.append(torch.from_numpy(flat).long().to(device))
    return torch.stack(rows, dim=0)
