#!/usr/bin/env bash
# setup_env.sh — Install Poetry and all CAAT dependencies in a local environment
# Usage: bash setup_env.sh [--gpu]

set -euo pipefail

GPU=false
for arg in "$@"; do
  case $arg in
    --gpu) GPU=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

if [ ! -d "env" ]; then
  echo "Creating bootstrap environment 'env'..."
  python3 -m venv env
fi
source env/bin/activate

echo "Installing Poetry into the current environment..."
pip install --upgrade pip
pip install poetry

echo "Configuring Poetry to use python3.11..."
poetry env use python3.11

echo "Installing core dependencies via Poetry..."
poetry install

echo "Installing AlphaFold/Colabfold extras..."
poetry install -E alphafold

if [ "$GPU" = true ]; then
  echo "Installing JAX CUDA libraries for GPU use..."
  poetry run pip install --no-warn-conflicts 'jax[cuda12]==0.4.28' jaxlib==0.4.28
else
  echo "Skipping GPU JAX install (pass --gpu to enable)"
fi

echo "--------------------------------------"
echo "Verifying installation..."
VENV_PYTHON=$(poetry env info --path)/bin/python
if $VENV_PYTHON -c "import colabfold; print('SUCCESS: colabfold is ready.')" &> /dev/null; then
    echo "Setup complete! CAAT environment is healthy."
    echo "To run jobs, use: poetry run python scripts/run_e2e_pipeline.py"
else
    echo "ERROR: Setup finished but colabfold is still not importable."
    exit 1
fi
