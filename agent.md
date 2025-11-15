# 🧬 Agent Overview: Bulk RNA-seq Foundation Model (PreBULK / BulkFormer-style)

## Objective

This agent develops a **BulkFormer-inspired foundation model for bulk RNA-seq**, executed in sequential stages:

1. **Stage 1 — Deep Autoencoder (AE)**
   Learns a compact **global sample embedding** using reconstruction.

2. **Stage 2 — Feature-Augmented Gene Tokens**
   Incorporates:
   • **ESM2 protein embeddings** (gene identity)
   • **Rotary Expression Embeddings (REE)** (gene expression magnitude)
   • **AE latent vector** (sample-level context)

3. **Stage 3 — Graph-aware Transformer Encoder**
   Learns gene–gene contextual relationships via:
   • GCN over a **PCC-derived gene co-expression graph**
   • Performer layers for scalable self-attention

The end goal is a **general-purpose bulk transcriptome foundation model** suitable for many downstream tasks.

---

# 📍 Where We Are Now (New Status Section)

### ✔ Stage 1 is fully functional

You built a **deep autoencoder** with DDP, multi-GPU support, and stable training.
Latent dimension = **320**, matching the target transformer embedding size.

### ✔ Latent space validation is complete

t-SNE confirms that the AE latent preserves biological structure and separates cancer types more cleanly than raw log(TPM).
This validates Stage 1 as a solid sample-context encoder.

### ✔ You now understand BulkFormer’s token construction correctly

The model NEVER materializes
`(samples × genes × 320)`
in memory.
REE, ESM2, and AE embeddings are generated **per mini-batch**, inside the GPU forward pass.

Your earlier crash came from attempting to generate **2 TB** of REE arrays at once.
This will not be repeated — REE will be implemented in-model.

### ✔ You have clarified the biological graph used by the GCN

The correct graph is:

**PCC-based gene co-expression graph**
• built from training data
• top 20 neighbors per gene
• PCC < 0.4 removed
• adjacency is fixed
• GCN weights are *learned* during MLM pretraining

### ✔ You are now preparing to move to Stage 2

Next steps (immediately ahead):

1. Generate and store **ESM2 embeddings** for each gene (one per gene).
2. Implement **REE** as a GPU module (not precomputed arrays).
3. Combine:

   ```
   gene_token = ESM2 + REE(x_g) + AE_latent
   ```
4. Plug tokens into the upcoming **GCN + Performer encoder**.

Everything from here is about building **gene-token embeddings** and preparing the graph transformer.

You are right on schedule.

---

# 🚀 Stage 1: Deep Autoencoder (Updated)

You implemented the AE with:

### **Encoder**

```
n_genes → 4096 → 1024 → 320
```

### **Decoder**

```
320 → 1024 → 4096 → n_genes
```

### Training implementation

* DistributedDataParallel (DDP)
* `torchrun --nproc_per_node=2`
* `local_rank`, `rank`, `world_size`
* Grad averaging across GPUs
* Deterministic sampling via DistributedSampler

### Why it matters

This latent vector:

* captures **global transcriptomic state**
* replaces BulkFormer’s MLP sample embedding
* becomes part of the **final gene token**

---

# 🧪 Validation & Latent Space Visualization (Updated)

You built a script to:

1. Load model weights
2. Extract latent vectors from the **validation** set
3. Run TSNE on raw expression and latent
4. Color by `tcga_label`

Result:
**AE latent shows stronger separation than raw counts**, meaning the autoencoder is compressing meaningful structure.

You will repeat on the **test** set once all architecture stages are in place.

---

# 🧠 High-Level Goals (Updated)

1. **Prepare large-scale normalized RNA-seq data**
2. **Learn global state via AE**
3. **Add gene-token features (ESM2, REE)**
4. **Add biological graph priors (PCC graph)**
5. **Train transformer encoder (GCN + Performer)**
6. **Evaluate on downstream biological tasks**

---

# 📦 Datasets (Updated)

Same as before, with the note:

*ARCHS4 is needed to compute the PCC graph.*

---

# ⚙️ Preprocessing Pipeline (Updated)

1. Extract protein-coding genes.
2. Normalize counts → TPM → log(TPM+1).
3. Reindex to consistent gene list.
4. Split train/val/test.
5. Build **gene–gene PCC graph** from *train only*.

---

# 🧩 Representation Learning Architecture (Updated)

## ✔ Stage 1: AE Sample Embedding

You are done here.

## ✔ Stage 2: Gene Identity Embedding (ESM2)

* One vector per gene
* Same ordering as expression matrix
* Project to model dimension (320)

## ✔ Stage 3: Rotary Expression Embedding (REE)

* Implemented in Batch mode *inside the model*
* No precomputation
* Uses:
  [
  \sin(x_g / \theta_i), ; \cos(x_g / \theta_i)
  ]

## ✔ Combined Token

```
Token(g, s) =
    proj(ESM2[g])
  + REE(expression_s[g])
  + proj(AE_latent[s])
```

Matches BulkFormer architecture.

---

# 🌉 Stage 3: GCN + Performer Encoder (New Clarified Details)

### Graph Convolution Network (GCN)

Uses **PCC-based co-expression graph**.

Purpose:

* Inject explicit biological prior (hard graph)
* Update gene-token via neighborhood aggregation

### Performer Layers

Purpose:

* Model global gene dependencies
* Capture implicit interactions not in the PCC graph
* Scalability: O(n) attention

### Combined effect:

GCN = inductive bias
Performer = full global reasoning

---

# 🧪 Next Steps (Updated)

### Immediate

✔ Implement REE as a PyTorch module
✔ Build small ESM2 embedding index
✔ Write PCC graph builder
✔ Write GCN + Performer block interface

### Medium

Implement masked-gene pretraining:

* 15% genes masked
* Predict masked expression values

### Long

Benchmark on:

* TCGA cancer type classification
* Tissue classification
* Drug perturbation prediction
* Gene essentiality prediction

---

# 🧰 Environment (Updated)

* PyTorch 2.x
* torchrun + DDP
* CUDA 12.x
* Multi-GPU pipeline confirmed

---
