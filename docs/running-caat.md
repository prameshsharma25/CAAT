# Usage

CAAT provides three entry points depending on your workflow needs.

## Option A: Full End-to-End Pipeline

Run structure prediction and attention analysis in one command. This is the recommended starting point for most users.
The end-to-end wrapper reruns prediction and overwrites matching result and
attention filenames so stale ColabFold completion markers cannot skip required
attention generation.
```bash
poetry run python scripts/run_e2e_pipeline.py \
  --query-seq-path <path/to/sequence.fasta> \
  --query-name <protein_name> \
  [OPTIONS]
```

### Prediction Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--query-seq-path` | - | Path to MSA or FASTA file |
| `--query-name` | - | **Required.** Identifier for your query protein (e.g., `XCL1`) |
| `--target-name` | `None` | Identifier for target/reference protein (for comparative analysis) |
| `--target-seq-path` | `None` | Path to target sequence file (for comparative analysis) |
| `--alignment-path` | `None` | Path to MSA alignment file (for comparative analysis) |
| `--model-type` | `alphafold2` | AlphaFold model variant to use |
| `--num-models` | `5` | Number of models to generate |
| `--result-dir` | `results` | Output directory for PDB structures |
| `--save-attention-npy` | `False` | Retain raw tensors selected for analysis; combine with `--include-extra-msa` to retain extra-MSA tensors |
| `--attention-output-dir` | `attention_outputs` | Directory for raw attention files |
| `--save-attention-compressed` | `False` | Save attention in compressed H5 format |
| `--save-intermediate-structures` | `None` | Directory for intermediate structure outputs |

### Analysis Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--vis-output-dir` | `visualizations` | Output directory for plots |
| `--query-highlight-indices` | `None` | Comma-separated residue positions to highlight (1-indexed, e.g., `1,5,10`) |
| `--target-highlight-indices` | `None` | Residue positions to highlight in target |
| `--query-highlight-color` | `#AE0639` | Hex color for query highlights |
| `--target-highlight-color` | `#1f77b4` | Hex color for target highlights |
| `--save-attention-npy` | `False` | Retain raw tensors selected for analysis; combine with `--include-extra-msa` to retain extra-MSA tensors |
| `--include-extra-msa` | `False` | Emit and analyze extra-MSA Evoformer attention blocks; by default only main blocks are written and analyzed |

---

## Option B: Generate Attention Heads Only

Extract attention weights without visualization. Useful for custom downstream analysis.
```bash
poetry run python scripts/run_attention_heads.py \
  --query-seq-path <path/to/sequence.fasta> \
  --query-name <protein_name> \
  [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--query-seq-path` | - | Path to input MSA (`.a3m`) or FASTA file |
| `--query-name` | - | **Required.** Protein identifier |
| `--model-type` | `alphafold2` | Model variant (e.g., `alphafold2_multimer_v3`) |
| `--attention-output-dir` | `attention_outputs` | Where to save `.npy` attention files |
| `--result-dir` | `results` | Directory for final PDB structures |
| `--num-models` | `5` | Number of models to run |
| `--save-attention-compressed` | `False` | Export compressed H5 format |
| `--include-extra-msa` | `False` | Emit extra-MSA Evoformer attention blocks in addition to main blocks |
| `--save-intermediate-structures` | `None` | Save intermediate evoformer structures |

---

## Option C: Using ColabFold Directly

CAAT extends ColabFold with custom attention output capabilities. Use the standard `colabfold_batch` command with additional flags:
```bash
poetry run colabfold_batch \
  <input> <results> \
  --attention-output-dir <path> \
  [STANDARD_COLABFOLD_OPTIONS]
```

### CAAT-Specific Flags

| Argument | Description |
|----------|-------------|
| `--attention-output-dir` | Directory to save attention head matrices (`.npy` files) |
| `--include-extra-msa` | Emit extra-MSA Evoformer attention blocks in addition to main blocks |
| `--save-intermediate-structures` | Directory to save intermediate evoformer structures |

For full ColabFold options, see the [ColabFold documentation](https://github.com/sokrypton/ColabFold) or run:
```bash
poetry run colabfold_batch --help
```

---

## Option D: Analysis Only

Visualize and compare pre-computed attention heads. Use this when you already have `.npy` attention files.
```bash
poetry run python scripts/run_analysis_pipeline.py \
  --query-attn-dir <path/to/attention_files> \
  --query-name <protein_name> \
  --query-seq-path <path/to/sequence.fasta> \
  [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--query-attn-dir` | - | **Required.** Directory containing `.npy` attention files for query |
| `--query-name` | - | **Required.** Identifier for the query protein |
| `--query-seq-path` | - | **Required.** Path to query sequence (`.a3m` or `.fasta`) |
| `--target-attn-dir` | `None` | Attention directory for target protein (for comparative analysis) |
| `--target-name` | `None` | Target protein identifier (for comparative analysis) |
| `--target-seq-path` | `None` | Target sequence file (for comparative analysis) |
| `--alignment-path` | `None` | Alignment file mapping query to target (for comparative analysis) |
| `--output-dir` | `attention_visualizations` | Output directory for plots |
| `--query-highlight-indices` | `None` | Residues to highlight in query (1-indexed) |
| `--target-highlight-indices` | `None` | Residues to highlight in target (1-indexed) |
| `--query-highlight-color` | `#AE0639` | Hex color for query highlights |
| `--target-highlight-color` | `#1f77b4` | Hex color for target highlights |
| `--save-attention-npy` | `False` | Retain raw tensors selected for analysis; combine with `--include-extra-msa` to retain extra-MSA tensors |
| `--include-extra-msa` | `False` | Include extra-MSA Evoformer blocks in attention analysis; by default only main Evoformer blocks are analyzed |
