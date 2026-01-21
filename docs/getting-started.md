## Installation

CAAT uses **Poetry** for dependency management. Follow these steps to set up your environment.

### 1. Clone the Repository
```bash
git clone https://github.com/prameshsharma25/CAAT.git
cd CAAT
```

### 2. Install Core Dependencies
```bash
poetry install
```

### 3. Install Optional Dependencies

#### AlphaFold Support

If you plan to use AlphaFold integration:
```bash
poetry install -E alphafold
```

#### GPU Support (CUDA)

For local GPU usage or HPC cluster deployment, install JAX with CUDA support:
```bash
poetry run pip install --no-warn-conflicts 'jax[cuda12]==0.4.28' jaxlib==0.4.28
```

Note: GPU support requires CUDA 12.x to be installed on your system.