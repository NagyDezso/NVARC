"""
Subclass TRM inner module to inject teacher proposal embeddings into ``z_H``.

See package docstring in ``hybrid_arc.__init__`` for thesis limitations.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)

from .z_h_seed import SeedMode, apply_z_h_seed, embed_prior_grid_tokens


class TinyRecursiveReasoningModel_ACTV1_Inner_ZHSeed(TinyRecursiveReasoningModel_ACTV1_Inner):
    """
    Same weights as ``TinyRecursiveReasoningModel_ACTV1_Inner``, with optional
    ``batch["y_prior_tokens"]`` of shape ``[B, seq_len]`` (``-1`` = skip position for
    per-row masking when stacking mixed batches).
    """

    def __init__(self, config, *, gamma: float = 0.1, seed_mode: SeedMode = "add") -> None:
        super().__init__(config)
        self.gamma = float(gamma)
        self.seed_mode: SeedMode = seed_mode

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L

        y_prior_tokens = batch.get("y_prior_tokens")
        if y_prior_tokens is not None and self.gamma > 0.0:
            valid = y_prior_tokens >= 0
            hint = embed_prior_grid_tokens(self, y_prior_tokens.clamp(min=0))
            z_H = apply_z_h_seed(
                z_H,
                hint,
                puzzle_emb_len=self.puzzle_emb_len,
                gamma=self.gamma,
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


def _module_device(module: nn.Module) -> torch.device:
    """Device of first parameter or buffer (inner may be buffer-heavy)."""
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def patch_act_model_inner(model: nn.Module, *, gamma: float, seed_mode: SeedMode) -> TinyRecursiveReasoningModel_ACTV1_Inner_ZHSeed:
    """
    Replace ``model.model.inner`` on an ``ACTLossHead``-wrapped TRM.

    ``model`` is typically ``train_state.model`` from ``eval-arc-k-10`` (``ACTLossHead``).
    """
    act = model
    trm = act.model  # TinyRecursiveReasoningModel_ACTV1
    cfg = trm.config
    device = _module_device(trm.inner)
    new_inner = TinyRecursiveReasoningModel_ACTV1_Inner_ZHSeed(cfg, gamma=gamma, seed_mode=seed_mode)
    new_inner.load_state_dict(trm.inner.state_dict(), strict=True)
    new_inner.to(device)
    trm.inner = new_inner
    return new_inner
