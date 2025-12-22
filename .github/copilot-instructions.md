# BioFM Copilot Instructions

## Project Overview

**BioFM** is a PyTorch-based deep learning model for learning gene embeddings from bulk RNA-seq transcriptomics data. It combines expression-based learning (masked language modeling) with biological priors (protein sequences via ESM2, gene-gene networks via GCN).

**Key Components:**
- **Data pipeline** (notebooks `0_*.ipynb` → `2_*.ipynb`): ARCHS4 cancer transcriptomics → normalized expression matrices
- **Gene embeddings** (`1a_protein_embeddings.ipynb`): ESM2 protein language model embeddings
- **Graph construction** (`3_knn_for_graph_edges.ipynb`): KNN-based gene coexpression networks
- **Model training** (`4_bulkformer_run.py`): DDP multi-GPU training with MLM masking
- **Analysis** (`5_*.ipynb`, `7_*.ipynb`): Feature extraction, clustering, TCGA analysis

---

## Architecture Essentials

### Data Flow Pipeline

```
ARCHS4 HDF5
    ↓ (0_cancer_subset.ipynb)
Cancer-labeled metadata + stratified train/val/test splits
    ↓ (1a_protein_embeddings.ipynb)
ESM2 embeddings [19357 genes × 320 dim] + safe protein list
    ↓ (2a/2b/2c_*.ipynb)
Normalized log(TPM) matrices [N samples × 19357 genes]
    ↓ (3_knn_for_graph_edges.ipynb)
KNN graph edges [2 × 387140] (top-20 coexpression neighbors)
    ↓ (4_bulkformer_run.py)
Trained model + checkpoints (PyTorch + SafeTensors)
```

### Model Architecture (GBFormer blocks)

Each GBFormer block implements three fusion strategies:
1. **GCN layer** (`torch_geometric.GCNConv`): Propagates info via gene-gene coexpression graph
2. **Binning + local Performer**: Groups genes by learned bin scores; each bin gets its own attention layer
3. **Global Performer**: Attends across all genes

Key insight: **Binning creates expression-regime-specific transformations** (housekeeping vs. tissue-specific genes occupy different "bins").

---

## Critical Conventions & Patterns

### 1. Gene Order & Alignment Consistency
- **All models expect exactly 19,357 genes in canonical order** defined by `./data/ensembl/filtered/safe_sequences.csv`
- Order is **NOT alphabetical**; determined by ESM2 embedding extraction
- Mismatch breaks inference. Always verify: `len(canonical_order) == 19357` and gene order matches ESM2 metadata

**Example:**
```python
# Load canonical order (from 1a_protein_embeddings.ipynb)
safe_path = "./data/ensembl/filtered/safe_sequences.csv"
safe_df = pd.read_csv(safe_path)
canonical_order = safe_df["gene_symbol"].tolist()
assert len(canonical_order) == 19357, "Gene count mismatch!"
```

### 2. Normalization Pipeline: Raw Counts → Model Input
**Order matters:**
1. **Raw counts** (from ARCHS4) → remove zero genes
2. **Filter to protein-coding** using `gene_biotype=="protein_coding"` from Ensembl FASTA
3. **Length-normalize**: Divide by exon length (kb) from `canonical_genes_with_exon_lengths_safe_sequences.csv`
4. **TPM scaling**: `(counts/length) / sum(all_counts/length) * 1e6`
5. **Log transform**: `log(TPM + 1)` (NOT log-fold-change)
6. **Standardize**: `(x - mean) / std` per sample (z-score, fillna=0)

**Critical detail:** Use **merged exon lengths** (from GENCODE), not protein AA length or genomic span. See `2a_test_set.ipynb` for exact pipeline.

### 3. Masked Language Modeling (MLM) Training
- **Masking ratio**: 15% of genes per sample (hardcoded in `BulkMLMDataset`)
- **Mask token**: -10 (special value used during training only)
- **Loss computation**: MSE **only on masked positions**, not all genes (see `4_bulkformer_run.py` line ~200)
- Input format: `(X_masked, X_true, mask_indices)` where X is [B × G] shape

### 4. Distributed Training (DDP)
- **Launch with**: `torchrun --nproc_per_node=2 4_bulkformer_run.py` (2 GPUs)
- **DistributedSampler** automatically splits data; no manual sharding
- **Batch size × world_size = global batch**: batch_size=64, 2 GPUs → 128 samples/step
- **Graph edges (edge_index) must be moved to GPU** before model init; use `edge_index.to(device)` before passing to BulkFormer

### 5. Model Checkpoint Format
Saves three formats:
- **`.pt`** (PyTorch): `torch.save(model.state_dict(), path)`
- **`.safetensors`** (HF-compatible): For Hugging Face Hub uploads
- **`config.json`**: Metadata (dim, gb_repeat, gene_length, etc.) for reproducibility

Load like:
```python
state_dict = torch.load("bulkformer_checkpoints/epoch_4.pt", map_location="cpu")
model = BulkFormer(**cfg)
model.load_state_dict(state_dict)
```

### 6. Feature Extraction (Post-Training)
- Request `repr_layers=[last_layer_idx]` to get hidden states from final GBFormer block
- Use `model(X, repr_layers=[k])` → returns `(predictions, {k: embeddings})`
- **Dynamic layer indexing**: `last_layer_idx = len(model.gb_formers) - 1` (not hardcoded `[2]`)
- Aggregation: max/mean/median pooling across genes → sample-level embedding

---

## Key File Organization

```
.
├── 0_cancer_subset.ipynb          # ARCHS4 → cancer labels + train/val/test splits
├── 1a_protein_embeddings.ipynb   # ESM2 gene embeddings (19357 × 320)
├── 1b_tpm_norm.ipynb              # Normalization reference
├── 2a_test_set.ipynb              # Test data preprocessing pipeline
├── 2b_val_set.ipynb               # Validation set processing
├── 2c_train_set.ipynb             # Training set (76K+ samples)
├── 3_knn_for_graph_edges.ipynb   # Gene-gene KNN graph construction
├── 4_bulkformer_run.py            # DDP training script (main entry point)
├── model/bulkformer.py            # Core model + GBFormer block
├── utils/extract_genes.ipynb      # Ensembl FASTA parsing
├── data/
│   ├── archs4/
│   │   ├── splits/                # {train,val,test}_metadata.csv
│   │   └── processed_short_proteins/  # Normalized matrices (parquet + numpy)
│   ├── embeddings/
│   │   └── esm2_t6_8M_UR50D_gene_embeddings.pt
│   ├── gencode/
│   │   └── canonical_genes_with_exon_lengths_safe_sequences.csv
│   └── ensembl/
│       ├── Homo_sapiens.GRCh38.pep.all.fa
│       └── filtered/safe_sequences.csv
└── graph/
    └── edge_index_top20.pt         # [2 × 387140] sparse adjacency
```

---

## Common Debugging Patterns

### Device Mismatch on GPU
**Error**: `RuntimeError: Expected all tensors to be on the same device`
**Fix**: Move edge_index and embeddings to device before model init:
```python
edge_index = edge_index.to(device)
gene_emb = gene_emb.to(device)
model = BulkFormer(..., graph=edge_index, gene_emb=gene_emb, ...)
```

### Gene Count Mismatch
**Error**: Shape mismatch between input and model expectations
**Cause**: Input has ≠19357 genes
**Fix**: Verify canonical gene order was used in preprocessing; check `processed_short_proteins/test_gene_order_short.csv`

### Metadata-Expression Alignment
**Error**: Sample indices don't match between metadata and expression matrix
**Fix**: Load metadata with same index as expression (both use file_id from ARCHS4); verify with `assert set(meta.index) >= set(expr.index)`

---

## Development Workflow

### Running Training
```bash
# Single-node, 2 GPUs
torchrun --nproc_per_node=2 4_bulkformer_run.py

# Outputs checkpoints to ./bulkformer_checkpoints/ + validation loss per epoch
```

### Adding Features
1. **New normalization**: Modify `2a_test_set.ipynb` pipeline; verify correlation with TCGA's ground-truth TPM
2. **New model blocks**: Extend `GBFormer` in `model/bulkformer.py`; increment `gb_repeat` in `model_params`
3. **New analysis**: Use `extract_feature(model, X, feature_type='transcriptome_level'|'gene_level')` from `5b_tcga_analysis.ipynb`

### Reproducibility
- Seed: Set in DistributedSampler (`seed=42`)
- Model config saved to JSON alongside checkpoint
- Gene order always from `safe_sequences.csv`

---

## External Dependencies
- **Data**: ARCHS4 HDF5, Ensembl GRCh38 FASTA
- **ML**: torch, torch_geometric, performer_pytorch, ESM-2 pretrained
- **Bio**: biopython (SeqIO), archs4py
- **Analysis**: pandas, numpy, scikit-learn, gseapy, UMAP, matplotlib

