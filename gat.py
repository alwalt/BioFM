#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from performer_pytorch import Performer
from torch_geometric.nn import GATv2Conv
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
DIM = 320
HEADS = 4
PERFORMER_LAYERS = 2
MASK_RATIO = 0.15
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10


# ============================================================
# REE MODULE
# ============================================================
class REE(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        half = dim // 2
        i = torch.arange(1, half + 1).float()
        theta = 100 ** (2 * i / dim)
        self.register_buffer("theta", theta)

    def forward(self, x):
        # x : [B,G]
        angles = x.unsqueeze(-1) * self.theta
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ============================================================
# DATASET
# ============================================================
class GeneDataset(Dataset):
    def __init__(self, X, esm2_emb, ae_latent):
        """
        X : [N,G] expression
        esm2_emb : [G,320]
        ae_latent : [N,320]  <-- sample-specific!
        """
        self.X = X
        self.esm2 = esm2_emb
        self.ae = ae_latent

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return idx so model can index AE latent properly
        return idx, torch.tensor(self.X[idx], dtype=torch.float32)


# ============================================================
# GAT + Performer Block
# ============================================================
class GATPerformerBlock(nn.Module):
    def __init__(self, dim=DIM, heads=HEADS, performer_layers=PERFORMER_LAYERS):
        super().__init__()

        self.gat = GATv2Conv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=0.1
        )

        dim_head = dim // heads
        self.performers = nn.ModuleList([
            Performer(
                dim=dim,
                depth=1,
                heads=heads,
                dim_head=dim_head,
                causal=False
            )
            for _ in range(performer_layers)
        ])

        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        """
        x: [B,G,dim]
        """
        B, G, D = x.shape

        # Run GAT on each sample independently
        out = []
        flat = x.reshape(B * G, D)

        for b in range(B):
            segment = flat[b*G:(b+1)*G]        # [G, dim]
            out.append(self.gat(segment, edge_index))

        x = torch.stack(out, dim=0)

        # Performer layers
        for layer in self.performers:
            x = layer(x)

        return self.ln(x)


# ============================================================
# FULL MODEL
# ============================================================
class BulkFormerGAT(nn.Module):
    def __init__(self, dim=DIM, num_genes=19357):
        super().__init__()

        self.dim = dim
        self.num_genes = num_genes

        self.ree = REE(dim)
        self.gene_proj = nn.Linear(dim, dim)
        self.ae_proj = nn.Linear(dim, dim)
        self.expr_proj = nn.Linear(dim, dim)

        self.block = GATPerformerBlock(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, idx, x_expr, esm2_emb, ae_latent, edge_index, mask=None):
        """
        idx: [B]
        x_expr: [B,G]
        esm2_emb: [G,320]
        ae_latent: [N_samples,320]
        """

        B, G = x_expr.shape

        # ===== 1) REE(x) =====
        ree = self.ree(x_expr)                     # [B,G,320]

        # ===== 2) gene identity ESM2 =====
        gene_emb = self.gene_proj(esm2_emb)        # [G,320]
        gene_emb = gene_emb.unsqueeze(0).expand(B, G, -1)

        # ===== 3) AE latent (sample-level context) =====
        sample_ctx = ae_latent[idx]                # [B,320]
        sample_ctx = self.ae_proj(sample_ctx)
        sample_ctx = sample_ctx.unsqueeze(1).expand(B, G, -1)

        # ===== 4) Combine =====
        x = ree + gene_emb + sample_ctx
        x = self.expr_proj(x)

        # ===== 5) Masking =====
        if mask is not None:
            x_expr = torch.where(mask, torch.full_like(x_expr, -10.0), x_expr)

        # ===== 6) GAT + Performer =====
        x = self.block(x, edge_index)

        # ===== 7) Decoder → prediction =====
        return self.head(x).squeeze(-1)


# ============================================================
# MASK FUNCTION
# ============================================================
def random_mask(x, ratio=MASK_RATIO):
    return (torch.rand_like(x) < ratio)


# ============================================================
# TRAIN FUNCTIONS
# ============================================================
def train(model, loader, esm2, ae_latent, edge_index, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"[TRAIN] Training for {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        total = 0
        t0 = time.time()

        for idx_batch, x_batch in loader:
            idx_batch = idx_batch.to(device)
            x_batch = x_batch.to(device)

            mask = random_mask(x_batch)
            target = x_batch.clone()

            pred = model(idx_batch, x_batch, esm2, ae_latent, edge_index, mask)

            loss = F.mse_loss(pred[mask], target[mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"  Epoch {epoch+1}/{EPOCHS} | loss={total/len(loader):.6f} | time={time.time()-t0:.1f}s")


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n============================================================")
    print("BulkFormer-GAT Training Script Started")
    print("============================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] Device: {device}")

    # Load expression
    X = pd.read_parquet("./data/archs4/processed_short_proteins/test_expr_logtpm_short.parquet")
    X = X.T.values.astype("float32")
    N, G = X.shape
    print(f"[DATA] Expression loaded: {X.shape}")

    # Load ESM2
    esm2_raw = torch.load("./data/embeddings/esm2_t6_8M_UR50D_gene_embeddings.pt")
    esm2 = esm2_raw["embeddings"].float().to(device)
    print(f"[DATA] ESM2 loaded: {esm2.shape}")

    # Load AE latent
    ae_np = torch.load("./data/embeddings/ae_gene_latents_320_test_set.pt", weights_only=False)
    ae_latent = torch.tensor(ae_np, dtype=torch.float32).to(device)
    print(f"[DATA] AE latent loaded: {ae_latent.shape}")

    # Load graph edges
    edge_index = torch.load("./graph/edge_index_top20.pt").to(device)
    print(f"[DATA] Graph edges: {edge_index.shape}")

    # Dataset
    ds = GeneDataset(X, esm2, ae_latent)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = BulkFormerGAT().to(device)
    print("[MODEL] Initialized")

    # Train
    train(model, dl, esm2, ae_latent, edge_index, device)

    # Save
    torch.save(model.state_dict(), "bulkformer_gat.pt")
    print("[SAVE] Model saved → bulkformer_gat.pt")


if __name__ == "__main__":
    main()
