#!/usr/bin/env bash
# setup_env.sh — Install Poetry and all CAAT dependencies in a local environment
# Usage: bash setup_env.sh [--gpu]

set -euo pipefail

GPU=false
usage() {
  echo "Usage: bash setup_env.sh [--gpu]"
  echo "  --gpu  Install a matching CUDA 12 JAX stack from CAAT's supported range."
}

for arg in "$@"; do
  case $arg in
    --gpu) GPU=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg"; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -e "$SCRIPT_DIR/.venv" ]; then
  if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ] || \
      ! "$SCRIPT_DIR/.venv/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "ERROR: The existing .venv does not contain a working Python 3.11+ interpreter." >&2
    echo "Move it aside (for example, 'mv .venv .venv.backup') and rerun this script." >&2
    exit 1
  fi
  CAAT_PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  CAAT_PYTHON=python3.11
elif command -v python3 >/dev/null 2>&1; then
  CAAT_PYTHON=python3
else
  echo "ERROR: Python 3.11 or newer is required."
  exit 1
fi

if ! "$CAAT_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "ERROR: $CAAT_PYTHON is older than Python 3.11."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating project environment '.venv'..."
  "$CAAT_PYTHON" -m venv .venv
fi
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
POETRY="$SCRIPT_DIR/.venv/bin/poetry"
export POETRY_VIRTUALENVS_IN_PROJECT=true

echo "Installing Poetry into the current environment..."
"$VENV_PYTHON" -m pip install --upgrade pip poetry

echo "Configuring Poetry to use the project environment..."
"$POETRY" env use "$VENV_PYTHON"

echo "Installing CAAT with AlphaFold support..."
"$POETRY" install -E alphafold

echo "Installing this checkout's bundled AlphaFold package..."
"$POETRY" run python -m pip install ./alphafold

if [ "$GPU" = true ]; then
  echo "Installing a matching CUDA 12 JAX stack..."
  "$POETRY" run python -m pip install --upgrade 'jax[cuda12]>=0.5.2,<0.11'
else
  echo "Skipping GPU JAX install (pass --gpu to enable)"
fi

echo "--------------------------------------"
echo "Verifying installation..."
"$POETRY" run python -c 'import alphafold, colabfold, jax; print("JAX:", jax.__version__); print("Devices:", jax.devices()); print("SUCCESS: CAAT imports are ready.")'

if [ "$GPU" = true ]; then
  "$POETRY" run python -c 'import jax; devices = jax.devices(); raise SystemExit(0 if any(d.platform == "gpu" for d in devices) else "ERROR: --gpu was requested, but JAX did not find a GPU. Check the CUDA 12 driver/runtime and cuDNN installation.")'
fi

echo "Setup complete! CAAT environment is healthy."
echo "To run jobs, use: poetry run python scripts/run_e2e_pipeline.py"
