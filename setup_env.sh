#!/usr/bin/env bash
# setup_env.sh — Install all CAAT dependencies
# Usage: bash setup_env.sh [--gpu]

set -euo pipefail

GPU=false
for arg in "$@"; do
  case $arg in
    --gpu) GPU=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

echo "Checking Python version..."
REQUIRED="3.11"

if command -v poetry &> /dev/null && poetry env info --path &> /dev/null; then
  PYTHON_BIN="$(poetry env info --path)/bin/python"
else
  PYTHON_BIN="python3"
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')

if [[ "$(echo -e "$PYTHON_VERSION\n$REQUIRED" | sort -V | head -1)" != "$REQUIRED" ]]; then
  echo "ERROR: Python $REQUIRED+ required, found $PYTHON_VERSION (via $PYTHON_BIN)"
  echo "Tip: Make sure Poetry is configured to use Python 3.11:"
  echo "  poetry env use /path/to/python3.11"
  exit 1
fi
echo "Python $PYTHON_VERSION OK (via $PYTHON_BIN)"

echo "Checking for Poetry..."
if ! command -v poetry &> /dev/null; then
  echo "Poetry not found. Installing..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "Poetry $(poetry --version) found"
fi

echo "Locking dependencies..."
poetry lock

echo "Installing core dependencies..."
poetry install

echo "Installing AlphaFold extras..."
poetry install -E alphafold

if [ "$GPU" = true ]; then
  echo "Installing JAX CUDA libraries for GPU/HPC use..."
  poetry run pip install --no-warn-conflicts 'jax[cuda12]==0.4.28' jaxlib==0.4.28
else
  echo "Skipping GPU JAX install (pass --gpu to enable)"
fi

echo ""
echo "Setup complete! Activate your environment with:"
echo "  poetry shell"
echo ""
echo "Or prefix commands with:"
echo "  poetry run python scripts/run_analysis_pipeline.py [OPTIONS]"
