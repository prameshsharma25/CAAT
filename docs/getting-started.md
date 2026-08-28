## Installation

CAAT uses **Poetry** for dependency management and requires Python 3.11 or
newer. The lock file records the dependency set tested by the repository, while
JAX itself is allowed to resolve anywhere in CAAT's supported range.

### 1. Clone the Repository
```bash
git clone https://github.com/prameshsharma25/CAAT.git
cd CAAT
```

### 2. Automated Setup

For CPU execution:

```bash
bash setup_env.sh
```

For an NVIDIA GPU with CUDA 12:

```bash
bash setup_env.sh --gpu
```

The setup script installs Poetry and CAAT's AlphaFold dependency group into the project environment. It builds and installs the bundled AlphaFold package as a normal wheel.

### 3. Manual Setup

If Poetry is already installed:

```bash
poetry install -E alphafold
poetry run python -m pip install ./alphafold
```

For CUDA 12, also install JAX's CUDA extra:

```bash
poetry run python -m pip install --upgrade 'jax[cuda12]>=0.5.2,<0.11'
```

CAAT no longer pins JAX to one patch release; the supported range is
`>=0.5.2,<0.11`. Installing `jax[cuda12]` in one command keeps JAX, jaxlib, and the CUDA plugin/PJRT packages on matching versions. The exact version selected by `poetry install` remains reproducible through `poetry.lock`.

Confirm the installation and detected device:

```bash
poetry run python -c "import jax; print(jax.__version__); print(jax.devices())"
```

GPU support requires a compatible NVIDIA driver and CUDA 12 runtime. Consult the official JAX [installation guide](https://docs.jax.dev/en/latest/installation.html) if JAX reports only a CPU device or cannot load CUDA/cuDNN.
