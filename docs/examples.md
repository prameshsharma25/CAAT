# XCL1 Example: End-to-End Pipeline

This example demonstrates a complete run of the E2E attention analysis pipeline using the XCL1 chemokine protein and its ancestral reconstruction Anc0. This comparison reveals evolutionarily significant attention patterns that may correspond to functional divergence between the modern and ancestral proteins.

## Running the Pipeline

### Command
```bash
poetry run python3 scripts/run_e2e_pipeline.py \
  --query-seq-path examples/XCL1/xcl1_seq.fa \
  --query-name XCL1 \
  --target-name Anc0 \
  --target-seq-path examples/XCL1/anc0_seq.fa \
  --alignment-path examples/XCL1/xcl1.fa
```

### Parameters Explained

- `--query-seq-path`: Path to the FASTA file containing the XCL1 sequence
- `--query-name`: Display name for the query protein (XCL1)
- `--target-name`: Display name for the target/reference protein (Anc0)
- `--target-seq-path`: Path to the FASTA file containing the Anc0 ancestral sequence
- `--alignment-path`: Path to the multiple sequence alignment file used for evolutionary context

Note: Requires GPU usage

## Results

The pipeline generates several output visualizations that provide complementary views of the attention landscape.

### Average Attention Maps

Average attention maps show the mean attention weights across all attention heads and layers for each position in the sequence. These maps reveal which residues the model considers most important globally.

#### XCL1 Average Attention

![XCL1 Average Attention](figures/XCL1/XCL1_average_attention.png)

This heatmap displays the average attention pattern for the modern XCL1 protein. Brighter regions indicate positions that receive higher attention weights, suggesting structural or functional importance.

#### Anc0 Average Attention

![Anc0 Average Attention](figures/XCL1/Anc0_average_attention.png)

The ancestral Anc0 protein's average attention pattern serves as the evolutionary baseline. Comparing this to XCL1 reveals how attention patterns have shifted over evolutionary time.

### Attention Difference Maps

The attention difference map is the **core analytical output** of this pipeline. It quantifies evolutionary divergence by computing the element-wise difference between modern and ancestral attention matrices.

**Calculation**: `Difference = Attention(XCL1) - Attention(Anc0)`

This subtraction highlights evolutionarily divergent "hotspots" where attention patterns have changed significantly.

#### XCL1 Attention Difference (Query Perspective)

![XCL1 Attention Difference](figures/XCL1/XCL1_attention_difference.png)

#### Anc0 Attention Difference (Target Perspective)

![Anc0 Attention Difference](figures/XCL1/Anc0_attention_difference.png)

The attention difference map from the ancestral protein's perspective provides the complementary view, highlighting which ancestral features have been retained or lost in the modern protein.
