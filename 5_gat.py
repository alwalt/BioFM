import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch_geometric.nn import GATv2Conv
from performer_pytorch import Performer
import pandas as pd
import numpy as np

# Device will be set later during DDP initialization
device = None


# ============================================================
# 1️⃣ ROTARY EXPRESSION EMBEDDING
# ============================================================
class PositionalExprEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mask_token_id = -10  # same as BulkFormer
        self.inv_freq = nn.Parameter(
            1.0 / (100 ** (torch.arange(0, dim, 2).float() / dim)),
            requires_grad=False
        )

    def forward(self, x):
        # x : [batch, genes]
        mask_idx = (x == self.mask_token_id).nonzero()

        rot = torch.einsum("bi,j->bij", x, self.inv_freq)
        rot = torch.cat((rot.sin(), rot.cos()), dim=-1)

        # zero out masked tokens
        if len(mask_idx) > 0:
            rot[mask_idx[:, 0], mask_idx[:, 1]] = 0

        return rot


# ============================================================
# 2️⃣ GBFormer (GAT + Performer blocks)
# ============================================================
class GBFormer(nn.Module):
    def __init__(self, dim, gene_length,
                 bin_head=2, full_head=2, bins=6, p_repeat=1):
        super().__init__()

        self.dim = dim
        self.bins = bins
        self.gene_length = gene_length

        # GAT module
        self.g = GATv2Conv(dim, dim, add_self_loops=False)

        # bin selector
        self.which_b = nn.Linear(dim, 1)

        # small Performer heads
        self.b = nn.ModuleList([
            Performer(
                dim=dim,
                heads=bin_head,
                dim_head=32,
                depth=1,
                attn_dropout=0.1,
                reversible=False
            )
            for _ in range(bins)
        ])

        self.f = nn.ModuleList([
            Performer(
                dim=dim,
                heads=full_head,
                dim_head=32,
                depth=1,
                attn_dropout=0.1,
                reversible=False
            )
            for _ in range(p_repeat)
        ])

        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        # x: [B, G, E]
        B, G, E = x.shape

        x = self.ln(x)

        # --- SAFE GRAPH UPDATE (no in-place) ---
        x_graph = []
        for b in range(B):
            gx = self.g(x[b], edge_index)
            x_graph.append(x[b] + gx)
        x = torch.stack(x_graph, dim=0)

        # --- choose bins ---
        scores = self.which_b(x).squeeze(-1)      # [B, G]
        order = torch.argsort(scores, dim=1, descending=True)  # [B, G]
        order_exp = order.unsqueeze(-1).expand(-1, -1, E)

        # reorder (no in-place)
        x_sorted = torch.gather(x, 1, order_exp)

        # split
        n = (G - 1) // self.bins + 1
        chunks = torch.split(x_sorted, n, dim=1)

        # run performer per chunk
        outs = []
        for chunk, layer in zip(chunks, self.b):
            outs.append(layer(chunk))
        xs = torch.cat(outs, dim=1)

        # --- UNSORT (no inplace scatter_) ---
        out = torch.zeros_like(xs)
        out = out.scatter(1, order_exp, xs)   # **THIS IS NOT INPLACE**

        # --- global Performer ---
        for layer in self.f:
            out = layer(out)

        return out




# ============================================================
# 3️⃣ BULKFORMER MODEL (Modified)
# ============================================================
class BulkFormer(nn.Module):
    def __init__(self, dim, graph, gene_emb, gene_length,
                 bin_head=2, full_head=2, bins=10, gb_repeat=2, p_repeat=1):
        super().__init__()

        self.dim = dim
        self.graph = graph
        self.gene_length = gene_length

        # 320-dim ESM2 embedding
        self.gene_emb = nn.Parameter(gene_emb)

        self.gene_emb_proj = nn.Sequential(
            nn.Linear(gene_emb.shape[1], 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        self.expr_emb = PositionalExprEmbedding(dim)

        # AE latent → sample context vector
        self.ae_enc = nn.Sequential(
            nn.Linear(gene_length, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        self.x_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        self.blocks = nn.ModuleList([
            GBFormer(dim, gene_length,
                     bin_head=bin_head,
                     full_head=full_head,
                     bins=bins,
                     p_repeat=p_repeat)
            for _ in range(gb_repeat)
        ])

        self.ln = nn.LayerNorm(dim)

        # Final head predicts expression
        self.head = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, 1)
        )

    def forward(self, x, ae_latent=None):
        # x: [B, G]
        B, G = x.shape

        gene_tok = self.gene_emb_proj(self.gene_emb).unsqueeze(0)  # [1, G, dim]
        expr_tok = self.expr_emb(x)                                # [B, G, dim]
        ae_tok = ae_latent.unsqueeze(1)                            # [B, 1, dim]

        x = expr_tok + gene_tok + ae_tok
        x = self.x_proj(x)

        for block in self.blocks:
            x = block(x, self.graph)

        x = self.ln(x)

        out = self.head(x).squeeze(-1)  # predict masked expression
        return out


# ============================================================
# 4️⃣ DATASET + MASKING
# ============================================================
class RNADataset(Dataset):
    def __init__(self, X, ae_latent, mask_ratio=0.15):
        self.X = X
        self.ae = ae_latent
        self.mask_ratio = mask_ratio

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        ae = self.ae[idx]

        # Mask 15% positions
        mask = torch.rand_like(x) < self.mask_ratio
        y = x.clone()
        x[mask] = -10  # mask token

        return x, y, ae


# ============================================================
# 5️⃣ INITIALIZE DEVICE & DDP
# ============================================================
script_start = time.time()
print("\n" + "="*70)
print("BulkFormer GBFormer Training Script")
print("="*70)

print("\n[SETUP] Initializing distributed training...")
init_start = time.time()

# Initialize DDP if running with torchrun
is_distributed = False
rank = 0

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # Check if environment variables are set (indicates torchrun execution)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            gpu_id = rank % torch.cuda.device_count()
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(device)
            is_distributed = True
            print(f"  ✓ DDP initialized: rank={rank}, world_size={world_size}, gpu={gpu_id}")
        except Exception as e:
            print(f"  ⚠ DDP init failed: {e}")
            print(f"  ℹ To use multi-GPU training, run: torchrun --nproc_per_node=2 gat2.py")
            is_distributed = False
            rank = 0
            device = "cuda:0"
    else:
        print(f"  ℹ Multi-GPU environment not detected.")
        print(f"  ℹ To use multi-GPU training, run: torchrun --nproc_per_node=2 gat2.py")
        print(f"  ℹ GPU count available: {torch.cuda.device_count()}")
        device = "cuda:0"
        is_distributed = False
else:
    print(f"  ℹ Single GPU mode (device count: {torch.cuda.device_count()})")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_distributed = False

print(f"[SETUP] Using device: {device}")

# ============================================================
# 6️⃣ LOAD DATA
# ============================================================
print("\n[DATA] Loading expression data...")
load_start = time.time()
X = pd.read_parquet("./data/archs4/processed_short_proteins/test_expr_logtpm_short.parquet")
X = torch.tensor(X.T.values.astype("float32")).to(device)
N, G = X.shape
load_time = time.time() - load_start
print(f"  ✓ Expression shape: {X.shape}, time: {load_time:.2f}s")

print("\n[DATA] Loading ESM2 protein embeddings...")
load_start = time.time()
esm2_raw = torch.load("./data/embeddings/esm2_t6_8M_UR50D_gene_embeddings.pt")
esm2 = esm2_raw["embeddings"].float().to(device)
load_time = time.time() - load_start
print(f"  ✓ ESM2 shape: {esm2.shape}, time: {load_time:.2f}s")

print("\n[DATA] Loading autoencoder latent embeddings...")
load_start = time.time()
ae_np = torch.load("./data/embeddings/ae_gene_latents_320_test_set.pt", weights_only=False)
ae_latent = torch.tensor(ae_np, dtype=torch.float32).to(device)
load_time = time.time() - load_start
print(f"  ✓ AE latent shape: {ae_latent.shape}, time: {load_time:.2f}s")

print("\n[DATA] Loading KNN gene graph edges...")
load_start = time.time()
edge_index = torch.load("./graph/edge_index_top20.pt").long().to(device)
load_time = time.time() - load_start
print(f"  ✓ Edge index shape: {edge_index.shape}, time: {load_time:.2f}s")



print("\n[DATA] Creating dataset and dataloader...")
ds_start = time.time()
dataset = RNADataset(X, ae_latent)

if is_distributed:
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), 
                                  rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0)
    print(f"  ✓ Created DistributedSampler: rank={rank}")
else:
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"  ✓ Created DataLoader (single GPU)")

ds_time = time.time() - ds_start
print(f"  ✓ Dataset: {len(dataset)} samples, {len(loader)} batches, time: {ds_time:.2f}s")

print("\n[MODEL] Initializing BulkFormer...")
model_start = time.time()
model = BulkFormer(
    dim=320,
    graph=edge_index,
    gene_emb=esm2,
    gene_length=G,
    gb_repeat=1,     ## only ONE BulkFormer block
    bins=1,          # ONE bin performer
    bin_head=2,
    full_head=2,
    p_repeat=1
).to(device)

print(model)


if is_distributed:
    gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
    model = DDP(model, device_ids=[gpu_idx], output_device=gpu_idx, find_unused_parameters=True)
    print(f"  ✓ Model wrapped with DDP")

model_time = time.time() - model_start
total_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model initialized with {total_params:,} parameters, time: {model_time:.2f}s")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

init_time = time.time() - init_start
print(f"\n  ✓ Total setup time: {init_time:.2f}s")


# ============================================================
# 7️⃣ TRAIN LOOP (with timing and progress tracking)
# ============================================================
print("\n[TRAIN] Starting training loop...")
train_start = time.time()
num_epochs = 5

for epoch in range(num_epochs):
    epoch_start = time.time()
    total_loss = 0.0
    num_batches = 0
    running_loss = 0.0
    
    if is_distributed and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    for batch_idx, (x, y, ae) in enumerate(loader):
        batch_start = time.time()
        
        x, y, ae = x.to(device), y.to(device), ae.to(device)
        
        optimizer.zero_grad()
        pred = model(x, ae)

        mask = (x == -10)
        loss = loss_fn(pred[mask], y[mask])

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        running_loss += loss_val
        num_batches += 1
        batch_time = time.time() - batch_start
        
        # Print progress every 25% of batches
        if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
            running_avg = running_loss / max(1, len(loader) // 4)
            print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss_val:.6f} | Avg: {running_avg:.6f} | Time: {batch_time:.3f}s")
            running_loss = 0.0

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    
    if rank == 0:  # Print only from rank 0 in distributed training
        print(f"\n  ╔════════════════════════════════════════════╗")
        print(f"  ║ Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")
        print(f"  ╚════════════════════════════════════════════╝\n")

train_time = time.time() - train_start

print("\n[SAVE] Saving model...")
save_start = time.time()
if is_distributed:
    torch.save(model.module.state_dict(), "bulkformer_gbformer.pt")
else:
    torch.save(model.state_dict(), "bulkformer_gbformer.pt")
save_time = time.time() - save_start
print(f"  ✓ Model saved to bulkformer_gbformer.pt, time: {save_time:.2f}s")

# Clean up DDP
if is_distributed:
    dist.destroy_process_group()
    print("  ✓ DDP cleanup complete")

total_time = time.time() - script_start
print("\n" + "="*70)
print(f"Training completed!")
print(f"  Training time: {train_time:.2f}s")
print(f"  Total script time: {total_time:.2f}s ({total_time/60:.1f}m)")
print("="*70 + "\n")
