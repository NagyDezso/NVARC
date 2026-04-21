"""Write arc2-llm-trm-z-h-seed-kaggle.ipynb from hybrid_arc/*.py (run from NVARC root)."""

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ha = root / "hybrid_arc"
    files = [
        "__init__.py",
        "arc_tokenize.py",
        "z_h_seed.py",
        "trm_z_h_seed.py",
        "llm_stage.py",
        "trm_stage.py",
        "pipeline.py",
        "run_local.py",
    ]
    cells: list = []
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ARC hybrid: autoregressive proposer + TRM (z_H seeding)\n",
                "\n",
                "Mirror of `hybrid_arc/` for Kaggle. Adjust `TRM_ROOT`, checkpoint, and data paths.\n",
                "\n",
                "**Priors JSON**: base puzzle id → list of test output grids (see `hybrid_arc/priors.example.json` in the repo).\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "from pathlib import Path\n",
                "TRM_ROOT = Path(\"/kaggle/input/trm-code/TinyRecursiveModels-main\")  # noqa: modify\n",
                "NVARC = Path(\"/kaggle/working\")\n",
                "for p in (TRM_ROOT, NVARC):\n",
                "    sys.path.insert(0, str(p))\n",
                "os.environ.setdefault(\"DISABLE_COMPILE\", \"1\")\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!uv pip install --no-deps --system hydra-core omegaconf pyyaml pydantic argdantic coolname tqdm adam-atan2-pytorch numba || true\n",
            ],
        }
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "(NVARC / \"hybrid_arc\").mkdir(parents=True, exist_ok=True)\n",
            ],
        }
    )
    for fn in files:
        body = (ha / fn).read_text(encoding="utf-8")
        src = f"%%writefile hybrid_arc/{fn}\n" + body
        if not src.endswith("\n"):
            src += "\n"
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            }
        )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "from hybrid_arc.pipeline import run_mode\n",
                "\n",
                "CKPT = Path(\"/kaggle/input/your-trm-checkpoint/step_100000\")  # noqa: modify\n",
                "DATA = Path(\"/kaggle/input/your-arc2test-aug-128\")\n",
                "overrides = [f'data_paths_test=[\\\"{DATA.as_posix()}\\\"]', 'global_batch_size=8']\n",
                "run_mode(\n",
                "    trm_root=TRM_ROOT,\n",
                "    overrides=overrides,\n",
                "    checkpoint=str(CKPT),\n",
                "    mode=\"trm_only\",\n",
                "    gamma=0.1,\n",
                "    seed_mode=\"add\",\n",
                "    priors_path=None,\n",
                "    csv_path=Path(\"/kaggle/working/ablation.csv\"),\n",
                "    data_dir_identifiers=DATA,\n",
                ")\n",
            ],
        }
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    out = root / "arc2-llm-trm-z-h-seed-kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
