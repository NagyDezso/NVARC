#!/usr/bin/env bash
# Pull the NVARC augmented-puzzles dataset from Kaggle so we can skip the
# SDG regeneration step. This already contains every subset that train_sft.py
# expects: arc2_training, arc2_evaluation6, mini, concept, rearc,
# nvarc_training, nvarc_full.
#
# Also pulls the ARC Prize 2025 competition data. Kaggle ships the ground-truth
# answers as standalone *_solutions.json files (the per-task ARC-AGI-2 repo
# does not), which is what run_inference.py --solutions expects:
#   arc-agi_{training,evaluation,test}_challenges.json  -> --tasks
#   arc-agi_{training,evaluation}_solutions.json        -> --solutions
#
# Requires: `uv sync --project HRM` (pins the kaggle CLI), a configured
# ~/.kaggle/kaggle.json, and acceptance of the competition rules at
# https://www.kaggle.com/competitions/arc-prize-2025/rules
#
# Usage:
#   bash HRM/download_data.sh                # default target: data/grids_v15
#   bash HRM/download_data.sh /path/to/dir   # custom target

set -euo pipefail

# Absolute paths so the kaggle CLI invocation survives the cd into $TARGET.
HRM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HRM_DIR/.." && pwd)"

TARGET="${1:-data/grids_v15}"
mkdir -p "$TARGET"

# ARC Prize competition that ships the standalone challenges/solutions JSONs.
COMPETITION="arc-prize-2025"
COMP_DIR="$REPO_DIR/data/$COMPETITION"

# Prefer the kaggle CLI from the HRM uv venv (where it's pinned in pyproject)
# so we don't depend on a system-wide install.
KAGGLE=(uv run --project "$HRM_DIR" kaggle)
if ! "${KAGGLE[@]}" --help >/dev/null 2>&1; then
    if command -v kaggle >/dev/null 2>&1; then
        KAGGLE=(kaggle)
    else
        echo "error: kaggle CLI not available." >&2
        echo "       Run 'uv sync --project HRM' first, or install system-wide." >&2
        exit 1
    fi
fi

echo "Downloading $COMPETITION competition data (challenges + solutions)..."
mkdir -p "$COMP_DIR"
"${KAGGLE[@]}" competitions download -c "$COMPETITION" -p "$COMP_DIR"
unzip -o "$COMP_DIR/$COMPETITION.zip" -d "$COMP_DIR"
rm -f "$COMP_DIR/$COMPETITION.zip"
echo "Competition data in $COMP_DIR:"
ls -la "$COMP_DIR"

cd "$TARGET"

echo
echo "Downloading sorokin/nvarc-augmented-puzzles (~3.2M puzzles, large)..."
"${KAGGLE[@]}" datasets download -d sorokin/nvarc-augmented-puzzles
unzip -o nvarc-augmented-puzzles.zip
rm -f nvarc-augmented-puzzles.zip

echo
echo "Done. Contents of $TARGET:"
ls -la

echo
echo "Next:"
echo "  uv run --project HRM python HRM/prepare_data.py --in_dir $TARGET --out_dir data/hrm_v1"
echo "  uv run --project HRM python HRM/run_inference.py \\"
echo "      --tasks $COMP_DIR/arc-agi_evaluation_challenges.json \\"
echo "      --solutions $COMP_DIR/arc-agi_evaluation_solutions.json --out submission.json"
