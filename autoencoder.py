#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# -----------------------------------
# 1. DATASET
# -----------------------------------
class ExpressionDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


# -----------------------------------
# 2. MODEL
# -----------------------------------
class BulkAE(nn.Module):
    def __init__(self, n_genes, latent_dim=320):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_genes)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# -----------------------------------
# 3. TRAINING LOOP
# -----------------------------------
def main():
    
    t_start = time.time()

    # ---------------------------------------------------------
    # Distributed setup
    # ---------------------------------------------------------
    t0 = time.time()
    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING SETUP")
    print("="*60)
    
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"✓ Distributed setup complete")
        print(f"  GPUs: {world_size}")
        print(f"  Time: {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    t0 = time.time()
    if local_rank == 0:
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
    
    TRAIN_PARQUET = "./data/archs4/processed_short_proteins/train_expr_logtpm_short.parquet"
    VAL_PARQUET   = "./data/archs4/processed_short_proteins/val_expr_logtpm_short.parquet"

    expr_df = pd.read_parquet(TRAIN_PARQUET)
    X = expr_df.T.astype(np.float32).values
    n_samples, n_genes = X.shape

    val_df = pd.read_parquet(VAL_PARQUET)
    X_val = val_df.T.astype(np.float32).values

    if local_rank == 0:
        print(f"✓ Data loaded")
        print(f"  Train: {n_samples} samples × {n_genes} genes")
        print(f"  Val:   {X_val.shape[0]} samples × {X_val.shape[1]} genes")
        print(f"  Time: {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # Datasets & Distributed Samplers
    # ---------------------------------------------------------
    t0 = time.time()
    if local_rank == 0:
        print("\n" + "="*60)
        print("CREATING DATALOADERS")
        print("="*60)
    
    dataset = ExpressionDataset(X)
    val_dataset = ExpressionDataset(X_val)

    train_sampler = DistributedSampler(dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    if local_rank == 0:
        print(f"✓ DataLoaders created")
        print(f"  Train batches: {len(loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Time: {time.time()-t0:.2f}s")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    t0 = time.time()
    if local_rank == 0:
        print("\n" + "="*60)
        print("INITIALIZING MODEL")
        print("="*60)
    
    model = BulkAE(n_genes, latent_dim=320).to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    if local_rank == 0:
        print(f"✓ Model initialized")
        print(f"  Input: {n_genes} genes")
        print(f"  Latent dim: 320")
        print(f"  Time: {time.time()-t0:.2f}s")

    EPOCHS = 20
    loss_history = []
    val_loss_history = []
    
    # Early stopping
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    if local_rank == 0:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
    
    for epoch in range(EPOCHS):
        t_epoch = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = criterion(x_hat, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Sync average loss across GPUs
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / world_size / len(loader)

        # ---------------- VAL ----------------
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, _ = model(batch)
                vloss = criterion(x_hat, batch)
                total_val += vloss.item()

        total_val_tensor = torch.tensor(total_val, device=device)
        dist.all_reduce(total_val_tensor, op=dist.ReduceOp.SUM)
        avg_val = total_val_tensor.item() / world_size / len(val_loader)

        if local_rank == 0:
            epoch_time = time.time() - t_epoch
            loss_history.append(avg_loss)
            val_loss_history.append(avg_val)
            
            # Early stopping check
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                print(f"[Epoch {epoch:2d}/{EPOCHS}] train_loss={avg_loss:.5f} | val_loss={avg_val:.5f} | {epoch_time:.2f}s ✓ (best)")
            else:
                patience_counter += 1
                print(f"[Epoch {epoch:2d}/{EPOCHS}] train_loss={avg_loss:.5f} | val_loss={avg_val:.5f} | {epoch_time:.2f}s ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                    break

    # ---------------------------------------------------------
    # Save results (rank 0 only)
    # ---------------------------------------------------------
    if local_rank == 0:
        t0 = time.time()
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Extract module from DDP wrapper
        model_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        
        # Save model weights
        torch.save(model_to_save.state_dict(), "autoencoder_weights.pt")
        print(f"✓ Model weights saved")
        print(f"  autoencoder_weights.pt")
        
        # Save loss histories
        np.save("loss_history.npy", np.array(loss_history))
        np.save("val_loss_history.npy", np.array(val_loss_history))
        
        print(f"✓ Loss histories saved")
        print(f"  loss_history.npy")
        print(f"  val_loss_history.npy")
        print(f"  Time: {time.time()-t0:.2f}s")
        
        # Plot loss curves
        t0 = time.time()
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label="Train Loss", linewidth=2)
        plt.plot(val_loss_history, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.title("Autoencoder Training & Validation Loss", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Plot saved")
        print(f"  loss_curves.png")
        print(f"  Time: {time.time()-t0:.2f}s")
        
        total_time = time.time() - t_start
        print("\n" + "="*60)
        print(f"TOTAL TIME: {total_time/60:.2f} minutes ({total_time:.0f}s)")
        print("="*60 + "\n")


    dist.destroy_process_group()


if __name__ == "__main__":
    main()
