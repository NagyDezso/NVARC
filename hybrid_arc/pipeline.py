"""
Orchestration: optional teacher priors, TRM eval with ``z_H`` seeding, CSV ablation logs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .arc_tokenize import canonical_grid_to_row_space
from .trm_stage import SeedMode, run_hybrid_eval


@dataclass
class PriorStore:
    """Load ``priors.json``: ``{ base_puzzle_id: [ per-test-index list of grids ] }``."""

    data: Dict[str, List[Any]]

    @classmethod
    def from_path(cls, path: Path) -> "PriorStore":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def get(self, augmented_name: str, test_flat_index: int = 0) -> Optional[np.ndarray]:
        from dataset.build_arc_dataset import PuzzleIdSeparator  # noqa: WPS433

        sep = PuzzleIdSeparator
        base = augmented_name.split(sep)[0] if sep in augmented_name else augmented_name
        if base not in self.data:
            return None
        grids = self.data[base]
        if test_flat_index < 0 or test_flat_index >= len(grids):
            return None
        g = grids[test_flat_index]
        if g is None:
            return None
        a = np.asarray(g, dtype=np.uint8)
        if a.ndim != 2:
            return None
        if sep in augmented_name:
            a = canonical_grid_to_row_space(a, augmented_name)
        return a


def make_prior_fn(store: Optional[PriorStore]) -> Optional[Callable[..., List[Optional[Any]]]]:
    if store is None:
        return None

    def _fn(batch, row_names: List[str]) -> List[Optional[Any]]:
        out: List[Optional[Any]] = []
        for name in row_names:
            out.append(store.get(name, 0))
        return out

    return _fn


def log_ablation_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_mode(
    *,
    trm_root: Path,
    overrides: List[str],
    checkpoint: str,
    mode: str,
    gamma: float,
    seed_mode: SeedMode,
    priors_path: Optional[Path],
    csv_path: Optional[Path],
    data_dir_identifiers: Optional[Path],
    smoke_train_batches: Optional[int] = None,
    smoke_skip_eval: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    ``mode``: ``trm_only`` | ``z_h_seed`` | ``z_h_seed_zero`` (gamma 0, patched inner).
    """
    store = PriorStore.from_path(priors_path) if priors_path and priors_path.exists() else None
    prior_fn = make_prior_fn(store)

    use_seed = mode in ("z_h_seed", "z_h_seed_zero")
    g = 0.0 if mode == "z_h_seed_zero" else float(gamma)
    metrics = run_hybrid_eval(
        trm_root=trm_root,
        config_overrides=overrides,
        load_checkpoint=checkpoint,
        gamma=g,
        seed_mode=seed_mode,
        use_z_h_seed=use_seed,
        prior_batch_fn=prior_fn,
        seed_first_step_only=True,
        data_dir_for_identifiers=data_dir_identifiers,
        smoke_train_batches=smoke_train_batches,
        smoke_skip_eval=smoke_skip_eval,
    )
    if csv_path is not None:
        flat: Dict[str, Any] = {"mode": mode, "gamma": g, "seed_mode": seed_mode}
        if metrics:
            for split, m in metrics.items():
                if isinstance(m, dict):
                    for k, v in m.items():
                        flat[f"{split}/{k}"] = float(v) if hasattr(v, "item") else v
                else:
                    # Flat metric value (evaluator returned {metric: value} directly)
                    flat[split] = float(m) if hasattr(m, "item") else m
        log_ablation_row(csv_path, flat)
    return metrics
