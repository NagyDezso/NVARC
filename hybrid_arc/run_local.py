"""
Local entrypoint for hybrid ARC eval (``z_H`` seeding + optional priors JSON).

Run from repository root::

    python -m hybrid_arc.run_local --trm-root TRM --checkpoint path/to/step_N \\
        --overrides data_paths_test=[path/to/arc2test-aug-128] '+load_checkpoint=...' \\
        --modes trm_only,z_h_seed --gamma 0.1 --seed-mode add \\
        --priors-json optional.json --ablation-csv results.csv

Requires CUDA (same as ``TRM/eval-arc-k-10.py``). Set ``DISABLE_COMPILE=1`` (default in code).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _prepend_sys_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    trm = root / "TRM"
    for p in (trm, root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid ARC: TRM + optional z_H seed from priors JSON")
    ap.add_argument("--trm-root", type=Path, default=Path("TRM"), help="Path to TRM package (contains eval-arc-k-10.py, config/)")
    ap.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file passed as load_checkpoint")
    ap.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        default=[],
        help='Hydra overrides, e.g. data_paths_test=["D:/data/arc2test-aug-128"]',
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="trm_only",
        help="Comma-separated: trm_only | z_h_seed | z_h_seed_zero",
    )
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--seed-mode", type=str, default="add", choices=("add", "replace_blend"))
    ap.add_argument("--priors-json", type=Path, default=None, help="Optional priors JSON for z_h_seed (see hybrid_arc.pipeline.PriorStore)")
    ap.add_argument("--ablation-csv", type=Path, default=None, help="Append one row per mode with aggregate metrics")
    ap.add_argument("--identifiers-dir", type=Path, default=None, help="Directory containing identifiers.json (default: first data_paths_test or data_paths)")

    args = ap.parse_args()
    _prepend_sys_path()

    def _has_load_checkpoint_override(ovr: list[str]) -> bool:
        for o in ovr:
            rest = o[1:] if o.startswith("+") else o
            if rest.startswith("load_checkpoint="):
                return True
        return False

    overrides = list(args.overrides)
    if args.checkpoint and not _has_load_checkpoint_override(overrides):
        overrides.append(f"+load_checkpoint={args.checkpoint}")

    from hybrid_arc.pipeline import run_mode
    from hybrid_arc.trm_stage import SeedMode

    sm: SeedMode = "replace_blend" if args.seed_mode == "replace_blend" else "add"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        print(f"=== mode={mode} ===")
        run_mode(
            trm_root=args.trm_root.resolve(),
            overrides=overrides,
            checkpoint=args.checkpoint,
            mode=mode,
            gamma=args.gamma,
            seed_mode=sm,
            priors_path=args.priors_json,
            csv_path=args.ablation_csv,
            data_dir_identifiers=args.identifiers_dir,
        )


if __name__ == "__main__":
    main()
