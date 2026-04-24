"""
Build embedding hints for ``z_H`` seeding from flat TRM label token ids.

Warm-start targets TRM's solution-track latent (``z_H`` after ``reset_carry``), in the
same subspace read by ``lm_head``. This is **not** the same as training-time teacher
forcing on discrete grids between ACT steps.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

SeedMode = Literal["add", "replace_blend"]


def embed_prior_grid_tokens(inner: nn.Module, y_prior_tokens: torch.Tensor) -> torch.Tensor:
    """
    ``y_prior_tokens``: ``[B, seq_len]`` int64 in TRM vocab (PAD=0, EOS=1, colors+2).
    Returns ``[B, seq_len, D]`` matched to the magnitude/positional treatment that
    ``TinyRecursiveReasoningModel_ACTV1_Inner._input_embeddings`` applies to grid
    tokens, so the hint lives in the same space the z-track has been trained to see:
    ``embed_scale * [0.707 * (embed_tokens + embed_pos_grid_slice)]`` for learned
    pos, or ``embed_scale * embed_tokens`` for rope.
    """
    emb = inner.embed_tokens(y_prior_tokens.to(torch.int32))
    if getattr(inner.config, "pos_encodings", None) == "learned":
        pos_table = inner.embed_pos.embedding_weight.to(emb.dtype)
        grid_pos = pos_table[inner.puzzle_emb_len :]
        emb = 0.707106781 * (emb + grid_pos)
    return inner.embed_scale * emb


def apply_z_h_seed(
    z_H: torch.Tensor,
    hint: torch.Tensor,
    *,
    puzzle_emb_len: int,
    gamma: float,
    seed_mode: SeedMode,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mutate ``z_H`` grid slice ``[:, puzzle_emb_len:, :]`` in-place.

    ``hint`` must match spatial size ``z_H[:, puzzle_emb_len:, :].shape``.

    ``valid_mask`` optional ``[B, seq_len]`` bool; False positions skip update (keeps
    initialized ``z_H`` there). If ``None``, all positions updated.
    """
    sl = slice(puzzle_emb_len, None)
    zg = z_H[:, sl, :]
    hg = hint
    if hg.shape != zg.shape:
        raise ValueError(f"hint shape {hg.shape} != z_H grid slice {zg.shape}")
    if valid_mask is None:
        m = torch.ones(zg.shape[:2], dtype=torch.bool, device=zg.device)
    else:
        m = valid_mask
    m3 = m.unsqueeze(-1)
    g = float(gamma)
    if seed_mode == "add":
        zg[:] = torch.where(m3, zg + g * hg, zg)
    elif seed_mode == "replace_blend":
        zg[:] = torch.where(m3, (1.0 - g) * zg + g * hg, zg)
    else:
        raise ValueError(seed_mode)
    return z_H
