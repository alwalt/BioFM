# 🧬 Agent Overview: Bulk RNA-seq Foundation Model (PreBULK / BulkFormer-style)

## Objective
This agent develops and experiments with a **transformer-based foundation model for bulk RNA-seq data**.  
Inspired by **BulkFormer**, it treats transcriptomic profiles as token sequences — each gene as a "token" with learned embeddings —  
to learn generalizable gene–gene dependencies and biological context across diverse conditions and tissues.

---

## 🧠 High-Level Goals
1. **Assemble a large, diverse bulk RNA-seq corpus** (e.g., from ARCHS4, TCGA).
2. **Standardize all data** (TPM normalization, log transform, z-score) into a consistent gene-by-sample matrix.
3. **Embed genes and samples** into high-dimensional continuous spaces:
   - **Gene identity embeddings** via **ESM2** (mean-pooled amino acid embeddings).
   - **Expression embeddings** via **Rotary Expression Embedding (REE)** — a variant of ROPE adapted to encode magnitude and continuity.
   - **Sample context embeddings** via an MLP summarizing overall transcriptomic state.
4. **Pretrain a transformer encoder** on masked-token or reconstruction objectives to learn biological representations.
5. **Fine-tune or probe** for downstream tasks:
   - Disease classification (e.g., TCGA cancer types)
   - Tissue or perturbation prediction
   - Embedding visualization (UMAP / t-SNE)

---

## 📦 Datasets

| Source | Description | Format |
|---------|--------------|--------|
| **ARCHS4** | ~128k human RNA-seq samples (raw counts) | `.h5` |
| **TCGA** | Cancer transcriptomes (33 types) | `.tsv` / `.h5` |
| (Optional) GEO subsets | For validation / cross-domain testing | — |

---

## ⚙️ Preprocessing Pipeline (`PreBULK`)
The goal is to ensure consistency, comparability, and inclusion of all protein-coding genes.

### Steps
1. **Extract protein-coding gene list**
   - From `Homo_sapiens.GRCh38.pep.all.fa`
   - Keep only `gene_biotype:protein_coding`
2. **Load & normalize expression**
   - Retrieve counts via `archs4py`
   - Filter low-quality samples (`min_nonzero_genes ≥ 14,000`)
   - TPM normalization  
     \[
     \text{TPM}_{g,s} = \frac{\text{Counts}_{g,s} / \text{Length}_g}{\sum_g (\text{Counts}_{g,s} / \text{Length}_g)} \times 10^6
     \]
   - Log-transform: `log(TPM + 1)`
   - Z-score standardization per gene
3. **Ensure full alignment**
   - Reindex to include *all* protein-coding genes (zero-pad missing)
4. **Export or batch-stream**
   - Split into `train`, `val`, and `test` sets (stratified by cancer type)

---

## 🔍 Representation Learning

### 1. Gene Identity (ESM2)
- Extract amino acid sequences for each gene.
- Embed via **ESM2**.
- Aggregate via **mean pooling**.
- Produces an `E_ESM2` matrix of shape `[N_genes, d_model]`.

### 2. Rotary Expression Embedding (REE)
Adapt **rotary position encoding (ROPE)** to continuous gene expression values:
\[
REE(x_g) = [\cos(x_g / f_i), \sin(x_g / f_i)]_{i=1}^{d/2}
\]
where \( f_i = 10^{(i / d) \cdot \text{scaling factor}} \).  
Encodes magnitude & continuity of expression without rank loss.

### 3. Sample Context Embedding (MLP)
- A small MLP compresses the entire transcriptomic vector into a global embedding `E_MLP`.
- Adds sample-level biological context (e.g., global activation states).

### 4. Combined Representation
Final token representation for gene *g* in sample *s*:
\[
E_{g,s} = E_{ESM2,g} + E_{REE,g,s} + E_{MLP,s}
\]

---

## 🧩 Model Architecture
- **Encoder-only transformer** (BERT-like)
- Input: gene-token embeddings per sample
- Objective: masked gene reconstruction / contrastive pretraining
- Output: learned gene and sample embeddings

---

## 📊 Visualization & Quality Control
To validate preprocessing and embeddings:
- **t-SNE / UMAP** projections (test set)
- Stratified color coding by `tcga_label`
- Compare separability of cancer classes and tissues
- Track runtime efficiency across subsets (e.g., 5 min for 12k test samples)

---

## 🧪 Next Steps
- ✅ Confirm pipeline on test set (ARCHS4)
- 🔜 Integrate TCGA for cancer-type classification benchmark
- 🔜 Implement REE transformation layer
- 🔜 Generate ESM2 gene embeddings and align by Ensembl ID
- 🔜 Pretrain transformer on masked-token objective
- 🔜 Evaluate on disease classification (weighted F1) and clustering quality

---

## 🧰 Environment
Recommended setup (conda, WSL2 or Linux):
```bash
conda create -n biofm python=3.10
conda activate biofm
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn biopython
pip install archs4py umap-learn torch esm
````

Optional GPU acceleration (RAPIDS):

```bash
conda install -c rapidsai -c nvidia -c conda-forge cuml cudf python=3.10 cuda-version=12.2
```

---

## 📁 Folder Structure

```
project_root/
│
├── data/
│   ├── archs4/
│   │   ├── human_gene_v2.5.h5
│   │   └── splits/
│   ├── ensembl/
│   │   └── Homo_sapiens.GRCh38.pep.all.fa
│   └── tcga/
│
├── scripts/
│   ├── prebulk_preprocessing.py
│   ├── visualize_umap_tsne.py
│   └── train_transformer.py
│
├── models/
│   └── bulkformer_transformer.py
│
└── agent.md   ← (this file)
```

---

## 🧩 References

* **BulkFormer: A Foundation Model for Transcriptomic Data**
  [bioRxiv, 2024]
* **ESM2 Protein Language Model (Meta AI, 2022)**

---

*Last updated: November 2025*

```
---