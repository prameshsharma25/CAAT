#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [ -x "$script_dir/.venv/bin/python" ] && \
    "$script_dir/.venv/bin/python" -c 'import alphafold, colabfold' >/dev/null 2>&1; then
  python_cmd="$script_dir/.venv/bin/python"
elif [ -x "$script_dir/env/bin/python" ] && \
    "$script_dir/env/bin/python" -c 'import alphafold, colabfold' >/dev/null 2>&1; then
  python_cmd="$script_dir/env/bin/python"
else
  echo "ERROR: A working CAAT environment was not found. Run 'bash setup_env.sh --gpu' first." >&2
  exit 1
fi

# Keep GPU execution and numeric precision consistent between runs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
export TF_DETERMINISTIC_OPS="${TF_DETERMINISTIC_OPS:-1}"
export TF_CUDNN_DETERMINISTIC="${TF_CUDNN_DETERMINISTIC:-1}"
caat_existing_xla_flags="${XLA_FLAGS:-}"
export XLA_FLAGS="${caat_existing_xla_flags:+${caat_existing_xla_flags} }--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

# Avoid reserving nearly all GPU memory before prediction begins.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

"$python_cmd" scripts/run_e2e_pipeline.py \
  --query-seq-path examples/XCL1/xcl1_seq.a3m \
  --query-name XCL1 \
  --random-seed 0 \
  --num-seeds 1 \
  "$@"
