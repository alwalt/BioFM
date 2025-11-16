import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from performer_pytorch import Performer

model_params = {
    'dim': 320, # embedding dimension (640 original BF model)
    "bins": 0,
    "gb_repeat": 1, # 3 original BF model
    "p_repeat": 4,
    'bin_head': 12,
    'full_head': 8,
    'gene_length': 19357 # 20010 genes in original data
}

# Rotary Expression Embedding (REE) layer.
# This is a FIXED (non-trainable) embedding that converts continuous
# gene expression values into sinusoidal rotation features.
# Implemented as an nn.Module so PyTorch can:
#   - register inv_freq as a buffer,
#   - move it to GPU with the model,
#   - include it in state_dict(),
#   - run dynamic computations per sample (masking + sin/cos transform).
# Not a learned embedding — purely a deterministic transformation.
class PositionalExprEmbedding(nn.Module):
      def __init__(self, dim):
        super().__init__()
        self.mask_token_id = -10
        # Even numbers because rotary embeddings use pairs of sin and cos
        # Divide by dim to scale the frequencies appropriately
        # Small exponent, large frequencies
        # nn.Parameter allows us to move the tensor to GPU and save with the model
        # requires_grad=False means we don't want to update this during training
        self.inv_freq = nn.Parameter(1. / (100 ** (torch.arange(0, dim, 2).float() /dim)), requires_grad=False)

        # forward is required for nn.Module
        # it's the function your model will use when called
        # it's what happns to x when the model is called
        def forward(self, x):
            # First it creates a boolean mask and outputs the indices where the mask is true
            # Then it gives the coordinates of the masked values 
            # .nonzero() returns the indices of non-zero elements in the tensor
            x_mask_idx = (x == self.mask_token_id).nonzero()

            # Multiply each element of x by every frequency in inv_freq.
            # Einsum expands x from shape (b, i) to (b, i, j) by pairing each feature
            # with all frequency values, producing a frequency-modulated version of x.
            x - torch.einsum("bi,j->bij", x, self.inv_freq)
            # Take the sin and cos of every element in x
            # concatenates along the last dimension
            x = torch.cat((x.sin(), x.cos()), dim=-1)
            
            # This sets masked positions to zero
            x[x_mask_idx[:,0], x_mask_idx[:,1]] = 0

            # The final output will have zeros where the mask was true
            return x
      
# The encoder block
# GB stand for Graph and Bins
class GBFormer(nn.Module):
    def __init__(self, dim, gene_length, bin_head=4, full_head=4, bins=10, p_repeat=1):
        super().__init__()

        self.dim = dim 
        self.gene_length = gene_length
        self.bins = bins # number of bins
        self.p_repeat = p_repeat # how many global Performers
        self.bin_head = bin_head  # heads for LOCAL attention
        self.full_head = full_head  # heads for GLOBAL attention

        # This is the GCN layer
        # Injects prior biological knowledge using gene-gene graph
        # Number of in channels and out channels are both dim (640 for original)
        # cached=True means it will cache the computation of the normalized adjacency matrix
        # add_self_loops=False means it won't add self-loops to the graph
        # GCNConv implements the Kipf & Welling (2016) Graph Convolution:
        # 
        #       H' = D^{-1/2} Â D^{-1/2} H W
        #
        # Where:
        #   Â  = adjacency matrix (with or without self-loops)
        #   D  = degree matrix (diagonal; D[i,i] = number of neighbors of node i)
        #   H  = current node features (embeddings)
        #   W  = learnable weight matrix
        #   H' = updated node features after message passing
        #
        # The normalization D^{-1/2} Â D^{-1/2} "sandwich" makes aggregation stable:
        #   - prevents high-degree nodes from dominating
        #   - ensures messages are scaled fairly
        #   - keeps propagation symmetric for undirected graphs
        self.g = GCNConv(dim, dim, cached=True, add_self_looops=False)
        # self.g = GATv2Conv(dim, dim, add_self_loops=False)

        # Learns a scalar "bin score" for each gene embedding.
        # Higher score → earlier in sort → assigned to earlier bins.
        self.which_b = nn.Sequential(
            nn.Linear(self.dim, 1),            
        )

        # Each bin gets processed by its own Performer layer
        # Creates bin-conditional local attention
        # different expression bins capture different regimes of genes:
        # - high-expression genes vs low-expression genes 
        # - housekeeping vs tissue-specific
        # - strongly vs weakly variable
        # So each bin gets its own local attention model.
        self.b = nn.ModuleList([
            # The Performer layer replaces softmax attention with FAVOR+ mechanism
            # Performer uses POSITIVE orthogonal random features:
            #
            #   φ(x) = exp(-||x||^2 / 2) * [exp(ω1^T x), ..., exp(ωm^T x)] / sqrt(m)
            #
            # Where softmax(QK^T)V  ≈  φ(Q)(φ(K)^T V)
            # This improves efficienty and goes from O(N^2) to O(N)
            # depth = layers of attention (1 here)
            # heads = number of attention heads
            # dim_head = dimension of each head
            Performer(dim=self.dim, heads=self.bin_head, 
                      depth=1, dim_head=self.dim//self.full_head, attn_dropout=0.2, ff_dropout=0.2)
                      for _ in range(self.p_repeat)
            ])
        
        # Global Performer block
        # Final Peformer attends across all genes, not just locally
        self.f = nn.Sequential(*[
            Performer(dim=self.dim, heads=self.full_head, 
                      depth=1, dim_head=self.dim//self.full_head)
        ])

        # Normalizes each gene's embedding vector to have mean 0 and variance 1
        # layernorm normalizes along the last dimension
        self.layernorm = nn.LayerNorm(self.dim)

    def forward(self, x, graph):
        b, g, e = x.shape

        # Normalize gene embeddings
        x = self.layernorm(x)

        # Runs GCN layer, include edge_index calculated from KNN, and adds residual connection (skiplink)
        # Updates each gene's embedding by aggregating info from its neighbors in the gene graph
        # Residual helps stabilize training and allows gradients to flow better
        # GCNConv only accepts [G, dim] inputs, but PyTorch handles batching by flattening the batch dimension
        x = x + self.g(x, graph)

        # Binning helps reduce computational load and capture local patterns
        if self.bins > 0: 
            # Sort genes into bins based on learned scores
            which_b = self.which_b(x).squeeze(-1) # [B, G, 1] -> [B, G] into scores
            order = torch.sort(which_b, dim=1, descending=True)[1] # [B, G] indices of sorted scores
            order = order.unsqueeze(-1).repeat(1, 1, e) # Track original positions for all embedding dims
            n = (g-1) // self.bins + 1 # number of genes per bin

            # Forward
            x = x.gather(1, order) # reorder genes by bin scores
            xs = torch.split(x, n, dim=1) # split into #n bins

            # We now do local attention within each bin
            # self.b is a list of Performer layers
            # xs is a batch of gene embeddings split into bins
            # zip pairs each bin with its corresponding Performer layer
            # Then runs each bin through its Performer
            xs = [
                layer(x)
                for x, layer in zip(xs, self.b)
            ]
            # Finally, we concatenate the processed bins back together
            xs = torch.cat(xs, dim=1) 

            x = torch.empty_like(xs)
            x = x.scatter_(1, order, xs) # restore original gene order

        x = self.f(x) # Global attention across all genes


# nn.Module is the base class for all neural network modules in PyTorch.
class BulkFormer(nn.Module):
    def __init__(self,
                dim,
                graph,
                gene_emb,
                gene_length,
                bin_head=4,
                full_head=4,
                bins=10,
                gb_repeat=3,
                p_repeat=1,
                ):

        # Calls the parent class's __init__ method 
        super().__init__()

        self.dim = dim 
        self.gene_length = gene_length

        # bins by expression (continuous)
        # The number of heads in an attention layer is not fixed, 
        # but a configurable hyperparameter that is typically determined by the task, 
        # # model size, and desired performance. A common approach is to divide the model's 
        # embedding dimension by the number of heads to get the dimension for each head. 
        # Some popular architectures like the BERT-base model use 12 heads.
        self.bins = bins 
        self.bin_head = bin_head
        self.full_head = full_head

        self.gb_repeat = gb_repeat
        self.p_repeat = 1

        self.graph = graph

        # Tensor subclass. When used with Modules, it is automatically registered as a parameter and helps
        # cache hidden states 
        # Says hey this is learnable.
        self.gene_emb = nn.Parameter(gene_emb)

        # Sequential container. Adding a small neural network. 
        # Project ESM2 gene embeddings into the model’s latent space.
        # This is the MLP to blow up the dimension, capture non-linearities, and project back down.
        self.gene_emb_proj = nn.Sequential(
            # Applies a linear transformation to the incoming data: y = xA^T + b
            # input dim: gene_emb.shape[1], output dim: 4 * self.dim
            # multiplying by 4 is common in transformer architectures to increase model capacity
            nn.Linear(self.gene_emb.shape[1], 4 * self.dim),
            # Relu is zero for negative inputs and linear for positive inputs
            # Used to introduce non-linearity
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim)
        )

        # This is the positional embedding for the expression values (REE)
        self.expr_emb = PositionalExprEmbedding(self.dim) # also the size of 640 in original BF model

        # Refines and mixes the fused embedding before it’s passed into: GCN, Performer, Binning, Global
        # Equivalent of the pre-attention feed-forward block in transformers
        self.x_proj =nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

        # Stack multiple GBFormer blocks.
        # Each GBFormer block contains:
        #   - LayerNorm
        #   - a GCNConv layer (graph-based gene update)
        #   - binning logic to group genes by learned importance
        #   - local Performer attention applied within each bin
        #   - global Performer attention across all genes
        #
        # Using nn.ModuleList allows us to keep N repeated blocks
        # (gb_repeat times), just like stacking Transformer layers.
        # Allows them to be moved to cuda or device automatically
        # Modules list are used to build repeated/stacked layers
        self.gb_formers = nn.ModuleList([
            GBFormer(self.dim, self.gene_length, self.bin_head, self.full_head,
            self.bins, self.p_repeat)
            for _ in range(self.gb_repeat)
        ])

        self.layernorm = nn.LayerNorm(self.dim)

        # Sample level embedding 
        # Captures global context of entire transcriptome
        # This is were we trained a AE offline to get sample-level embeddings
        self.ae_enc = nn.Sequential( 
            # Compresses the gene expression vector into a lower-dimensional latent space
            nn.Linear(self.gene_length, 4 * self.dim), 
            nn.ReLU(), 
            # Projects the compressed representation into the final latent space of size dim
            nn.Linear(4 * self.dim, self.dim),
            nn.ReLU(),
        )

        # Final prediction head.
        # Takes each gene’s contextual embedding (dim)
        # → expands to 4*dim → non-linearity → collapses to 1 scalar.
        # Output shape: [B, G, 1] = predicted gene expression values.
        # Final ReLU enforces non-negative predictions.
        self.head = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.ReLU(),
            nn.Linear(4 * self.dim, 1),
            nn.ReLU(),
        )

    def forward(self, x, repr_layers=None):
        # This is the data a user provides notice shape is [b, g]
        b, g = x.shape

        # Here we add the REE, gene identity, and AE latent embeddings
        x = self.expr_emb(x) + self.gene_emb_proj(self.gene_emb) + self.ae_enc(x).unsqueeze(1)
        # Apply a feed-forward mixing layer (dim -> 4*dim -> dim)
        x = self.x_proj(x)

        hidden = {}
        for idx, layer, in enumerate(self.gb_formers):
            x = layer(x, self.graph)
            # repr_layers: optional list of block indices whose hidden states
            # should be returned. Useful for extracting intermediate layer outputs
            # for probing, interpretability, or downstream tasks.
            if repr_layers and idx in repr_layers:
                hidden[idx] = x

        x = self.layernorm(x)
        if repr_layers and idx in repr_layers:
            hidden[idx] = x

        x = self.head(x).squeeze(-1)

        if repr_layers:
            return x, hidden
        else:
            return x






