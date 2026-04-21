"""
hybrid_arc: autoregressive teacher proposal + TRM inference with optional z_H warm-start.

Thesis-oriented method (see module docstrings in ``z_h_seed``, ``trm_z_h_seed``):

- **Core idea**: embed a prior output grid ``y_prior`` with the same ARC token layout as
  TRM labels, then add or blend those embeddings into ``z_H`` on grid positions (after
  ``puzzle_emb_len``) before the recurrent ``L_level`` blocks. This targets TRM's
  solution-track latent (paper ``y \\approx z_H``), not discrete teacher-forcing of
  logits between ACT steps.

**Limitations** (ablate and report):

1. **OOD seeding**: checkpoints were trained with ``H_init`` / ``L_init`` latents, not
   teacher-derived slices; use modest ``gamma`` and report sensitivity.
2. **``z_L`` not seeded**: wrong macro proposals can waste depth reconciling ``z_H``.
3. **Injection timing**: default seeds only when ``y_prior_tokens`` is present on the
   first ACT step (caller removes the key afterward if configured).
4. **Augmentation alignment**: ``y_prior`` must live in the same dihedral/color frame as
   the TRM dataloader row (use ``arc_tokenize.apply_row_augmentation`` from canonical).
"""

__all__ = ["arc_tokenize", "z_h_seed", "trm_z_h_seed", "llm_stage", "trm_stage", "pipeline", "run_local"]
