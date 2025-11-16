import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠ safetensors not installed. Install with: pip install safetensors")

from bulkformer import BulkFormer, model_params  


# ===============================================================
# 1. Dataset with MLM-style masking
# ===============================================================
class BulkMLMDataset(Dataset):
    def __init__(self, X_np, mask_ratio=0.15, mask_token=-10):
        self.X = X_np.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        # mask 15% of genes
        g = x.shape[0]
        num_mask = int(g * self.mask_ratio)

        mask_idx = np.random.choice(g, num_mask, replace=False)
        x_masked = x.copy()
        x_masked[mask_idx] = self.mask_token

        return (
            torch.tensor(x_masked),
            torch.tensor(x),
            torch.tensor(mask_idx)
        )


# ===============================================================
# 2. Main training function
# ===============================================================
def main():
    script_start = time.time()
    print("\n" + "="*70)
    print("BulkFormer Training with GPU Optimization")
    print("="*70)

    device = "cuda"
    print(f"\n[SETUP] Device: {device}")

    # -----------------------------------------------------------
    # Load expression data
    # -----------------------------------------------------------
    print("\n[DATA] Loading expression data...")
    t_start = time.time()
    X_df = pd.read_parquet("./data/archs4/processed_short_proteins/test_expr_logtpm_short.parquet")
    X_df = X_df.T
    X_np = X_df.values   # shape [N, G]
    num_samples, num_genes = X_np.shape
    t_load = time.time() - t_start
    print(f"  ✓ Shape: {X_np.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Load ESM2 gene identity embeddings
    # -----------------------------------------------------------
    print("\n[DATA] Loading ESM2 gene embeddings...")
    t_start = time.time()
    esm2_data = torch.load("./data/embeddings/esm2_t6_8M_UR50D_gene_embeddings.pt")
    esm2_raw = esm2_data['embeddings'].float().to(device)
    t_load = time.time() - t_start
    print(f"  ✓ Shape: {esm2_raw.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Load gene–gene graph
    # -----------------------------------------------------------
    print("\n[DATA] Loading gene graph edges...")
    t_start = time.time()
    edge_index = torch.load("./graph/edge_index_top20.pt").long().to(device)
    t_load = time.time() - t_start
    print(f"  ✓ Shape: {edge_index.shape}, Time: {t_load:.2f}s")

    # -----------------------------------------------------------
    # Configure model parameters dynamically
    # -----------------------------------------------------------
    print("\n[MODEL] Configuring model parameters...")
    model_params["graph"] = edge_index
    model_params["gene_emb"] = esm2_raw
    model_params["gene_length"] = num_genes

    # -----------------------------------------------------------
    # Build model
    # -----------------------------------------------------------
    print("\n[MODEL] Initializing BulkFormer...")
    t_start = time.time()
    model = BulkFormer(**model_params).to(device)
    t_init = time.time() - t_start
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parameters: {total_params:,}, Init time: {t_init:.2f}s")

    # -----------------------------------------------------------
    # Prepare dataset
    # -----------------------------------------------------------
    print("\n[DATA] Creating dataset and dataloader...")
    t_start = time.time()
    dataset = BulkMLMDataset(X_np)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    t_prep = time.time() - t_start
    print(f"  ✓ Dataset size: {len(dataset)}, Batches: {len(loader)}, Time: {t_prep:.2f}s")

    # -----------------------------------------------------------
    # Optimizer & loss
    # -----------------------------------------------------------
    print("\n[OPTIM] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    print(f"  ✓ AdamW optimizer ready (lr=1e-4)")

    # -----------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------
    print("\n[TRAIN] Starting training loop...")
    print("="*70 + "\n")
    epochs = 5

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_masked, x_true, mask_idx) in enumerate(loader):
            batch_start = time.time()
            
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)

            # Forward pass
            pred = model(x_masked)    # shape [B, G]

            # Compute reconstruction loss *only at masked positions*
            loss_list = []
            for i in range(len(mask_idx)):
                idxs = mask_idx[i]
                loss_list.append(mse_loss(pred[i, idxs], x_true[i, idxs]))

            loss = torch.stack(loss_list).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            num_batches += 1
            batch_time = time.time() - batch_start
            
            # Print per-batch progress every 25% of batches
            if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
                avg_so_far = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss_val:.6f} | Avg: {avg_so_far:.6f} | Time: {batch_time:.3f}s")

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = running_loss / num_batches
        
        print(f"\n  ╔════════════════════════════════════════════╗")
        print(f"  ║ Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_avg_loss:.6f} | Time: {epoch_time:.2f}s")
        print(f"  ╚════════════════════════════════════════════╝\n")

    # -----------------------------------------------------------
    # Save checkpoint
    # -----------------------------------------------------------
    print("\n[SAVE] Saving model checkpoints...")
    save_start = time.time()
    
    # Create checkpoint directory
    ckpt_dir = Path(f"bulkformer_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    # 1. PyTorch native format (.pt)
    pt_path = ckpt_dir / f"epoch_{epoch}.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"  ✓ PyTorch format: {pt_path}")
    
    # 2. Hugging Face SafeTensors format (.safetensors) - recommended for HF Hub
    if HAS_SAFETENSORS:
        # Convert state_dict to CPU tensors for safetensors
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        safetensors_path = ckpt_dir / f"epoch_{epoch}.safetensors"
        save_safetensors(state_dict_cpu, str(safetensors_path))
        print(f"  ✓ SafeTensors format (HF-compatible): {safetensors_path}")
    
    # 3. Model config JSON for HF Hub
    config = {
        "model_type": "bulkformer",
        "num_genes": num_genes,
        "dim": model_params.get("dim", 320),
        "gb_repeat": model_params.get("gb_repeat", 2),
        "bins": model_params.get("bins", 10),
        "bin_head": model_params.get("bin_head", 2),
        "full_head": model_params.get("full_head", 2),
        "p_repeat": model_params.get("p_repeat", 1),
        "training_epoch": epoch,
        "final_loss": epoch_avg_loss
    }
    config_path = ckpt_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config: {config_path}")
    
    save_time = time.time() - save_start
    print(f"\n  ✓ All checkpoints saved in {ckpt_dir}/, Time: {save_time:.2f}s")
    print(f"\n  📦 To upload to Hugging Face Hub:")
    print(f"     huggingface-cli upload <username>/<repo-name> bulkformer_checkpoints/")
    

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    total_time = time.time() - script_start
    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"  Total script time: {total_time:.2f}s ({total_time/60:.1f}m)")
    print("="*70 + "\n")


# ===============================================================
# Run
# ===============================================================
if __name__ == "__main__":
    main()
