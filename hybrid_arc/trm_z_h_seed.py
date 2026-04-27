"""
Inject teacher-proposal embeddings into TRM's ``z_H`` by monkey-patching ``forward``
on the existing inner module — preserves parameter identities so the pre-built
optimizer + EMA references stay valid (a subclass-and-replace strategy silently
orphans them, which manifests as catastrophic eval regression at gamma=0).
"""

from __future__ import annotations

import types
from typing import Dict, Tuple

import torch
from torch import nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)

from .z_h_seed import SeedMode, apply_z_h_seed, embed_prior_grid_tokens


def _seeded_forward(
    self: TinyRecursiveReasoningModel_ACTV1_Inner,
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    batch: Dict[str, torch.Tensor],
) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Same iteration as ``TinyRecursiveReasoningModel_ACTV1_Inner.forward`` plus a
    one-shot z_H additive/blend seed when ``batch['y_prior_tokens']`` is present.

    Reads ``self.gamma`` and ``self.seed_mode`` (set by ``patch_act_model_inner``).
    """
    seq_info = dict(
        cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
    )
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    z_H, z_L = carry.z_H, carry.z_L

    y_prior_tokens = batch.get("y_prior_tokens")
    if y_prior_tokens is not None and float(self.gamma) > 0.0:
        valid = y_prior_tokens >= 0
        hint = embed_prior_grid_tokens(self, y_prior_tokens.clamp(min=0))
        # Match the magnitude of `_input_embeddings`: tokens go through `embed_scale`
        # there, so without this the z_H delta is ~sqrt(D)x too small.
        hint = hint * self.embed_scale
        z_H = apply_z_h_seed(
            z_H,
            hint,
            puzzle_emb_len=self.puzzle_emb_len,
            gamma=float(self.gamma),
            seed_mode=self.seed_mode,
            valid_mask=valid,
        )

    with torch.no_grad():
        for _H_step in range(self.config.H_cycles - 1):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
    for _L_step in range(self.config.L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
    z_H = self.L_level(z_H, z_L, **seq_info)

    new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
    q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


def patch_act_model_inner(model: nn.Module, *, gamma: float, seed_mode: SeedMode) -> nn.Module:
    """
    Monkey-patch ``model.model.inner.forward`` in place. Does NOT swap the module,
    so all parameter/buffer identities (and therefore the optimizer + EMA refs that
    were captured in ``init_train_state``) remain valid.

    ``model`` is ``train_state.model`` (``ACTLossHead``-wrapped TRM).
    """
    inner = model.model.inner  # TinyRecursiveReasoningModel_ACTV1_Inner
    inner.gamma = float(gamma)
    inner.seed_mode = seed_mode
    inner.forward = types.MethodType(_seeded_forward, inner)
    return inner


# Backwards-compat alias for any external imports / tests.
TinyRecursiveReasoningModel_ACTV1_Inner_ZHSeed = TinyRecursiveReasoningModel_ACTV1_Inner
