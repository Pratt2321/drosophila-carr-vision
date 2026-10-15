#!/usr/bin/env bash
# ============================================================
#  Drosophila Compound-Eye Vision Pipeline — Runner
# ============================================================
#  Installs dependencies into a local venv and runs the
#  simulation pipeline.  All CLI arguments are forwarded
#  directly to simulate_vision.py.
#
#  Usage:
#    ./run.sh                                          # defaults
#    ./run.sh --start-sec 140 --count 500              # custom window
#    ./run.sh --url "https://youtu.be/<ID>"            # YouTube video
#    ./run.sh --video other_video.mp4                  # local file
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON=""

# ------------------------------------------------------------------
# 1. Locate a suitable Python (>=3.12)
# ------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🪰  Drosophila Compound-Eye Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "[ERROR] No suitable Python found (need >=3.12)."
    echo "        Install Python 3.12+ and try again."
    exit 1
fi

echo "[1/4] Using $PYTHON ($($PYTHON --version 2>&1))"

# ------------------------------------------------------------------
# 2. Create / activate virtual environment
# ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[2/4] Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual environment already exists."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------
# 3. Install dependencies
# ------------------------------------------------------------------
echo "[3/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet 2>&1 | tail -5
echo "      ✓ Dependencies installed."

# ------------------------------------------------------------------
# 4. Run the pipeline — forward all CLI args
# ------------------------------------------------------------------
echo "[4/4] Running simulation pipeline..."
echo ""

python simulate_vision.py "$@"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Pipeline complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
