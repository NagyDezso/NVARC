#!/usr/bin/env bash
# Pull the NVARC augmented-puzzles dataset from Kaggle so we can skip the
# SDG regeneration step. This already contains every subset that train_sft.py
# expects: arc2_training, arc2_evaluation6, mini, concept, rearc,
# nvarc_training, nvarc_full.
#
# Requires: `pip install kaggle` and a configured ~/.kaggle/kaggle.json.
#
# Usage:
#   bash HRM/download_data.sh                # default target: data/grids_v15
#   bash HRM/download_data.sh /path/to/dir   # custom target

set -euo pipefail

TARGET="${1:-data/grids_v15}"
mkdir -p "$TARGET"

cd "$TARGET"

# Prefer the kaggle CLI from the HRM uv venv (where it's pinned in pyproject)
# so we don't depend on a system-wide install.
KAGGLE=(uv run --project HRM kaggle)
if ! "${KAGGLE[@]}" --help >/dev/null 2>&1; then
    if command -v kaggle >/dev/null 2>&1; then
        KAGGLE=(kaggle)
    else
        echo "error: kaggle CLI not available." >&2
        echo "       Run 'uv sync --project HRM' first, or install system-wide." >&2
        exit 1
    fi
fi

echo "Downloading sorokin/nvarc-augmented-puzzles (~3.2M puzzles, large)..."
"${KAGGLE[@]}" datasets download -d sorokin/nvarc-augmented-puzzles
unzip -o nvarc-augmented-puzzles.zip
rm -f nvarc-augmented-puzzles.zip

echo
echo "Done. Contents of $TARGET:"
ls -la

echo
echo "Next: python HRM/prepare_data.py --in_dir $TARGET --out_dir data/hrm_v1"
