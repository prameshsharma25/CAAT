#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

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

has_caat_gpu() {
  local candidate="$1"
  [ -x "$candidate" ] && \
    "$candidate" -c 'import alphafold, colabfold, jax; raise SystemExit(0 if any(device.platform == "gpu" for device in jax.devices()) else 1)' \
      >/dev/null 2>&1
}

if has_caat_gpu "$script_dir/.venv/bin/python"; then
  python_cmd="$script_dir/.venv/bin/python"
elif has_caat_gpu "$script_dir/env/bin/python"; then
  python_cmd="$script_dir/env/bin/python"
else
  echo "ERROR: No CAAT environment with a usable JAX GPU was found." >&2
  echo "Run 'bash setup_env.sh --gpu' on a GPU node, then retry." >&2
  exit 1
fi

"$python_cmd" scripts/run_e2e_pipeline.py \
  --query-seq-path examples/XCL1/xcl1_seq.a3m \
  --query-name XCL1 \
  --random-seed 0 \
  --num-seeds 1 \
  "$@"
