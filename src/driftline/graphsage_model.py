"""GraphSAGE over the entity graph: transaction nodes get a learned linear projection of real
numeric features, value nodes get a learned embedding (they have no intrinsic features -- they're
anonymous identity anchors). Both project into the same hidden_dim so message passing (SAGEConv)
can mix them.

Standard PyG mini-batch pattern: NeighborLoader samples a subgraph and returns `n_id`, the global
node ids of every node in the sampled subgraph, in an order where the first `batch_size` entries
are always the seed (target) nodes. The model gathers real features for txn-node ids and
embedding rows for value-node ids using that global id, rather than requiring PyG to carry
heterogeneous raw features through its sampler directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EntitySAGE(nn.Module):
    def __init__(self, num_numeric_features: int, num_value_nodes: int, hidden_dim: int = 32):
        super().__init__()
        self.num_value_nodes = num_value_nodes
        self.txn_proj = nn.Linear(num_numeric_features, hidden_dim)
        self.value_embedding = nn.Embedding(num_value_nodes, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def initial_features(self, n_id: torch.Tensor, txn_features_full: torch.Tensor, num_txn: int) -> torch.Tensor:
        is_txn = n_id < num_txn
        x = torch.zeros(len(n_id), self.txn_proj.out_features, device=n_id.device)
        if is_txn.any():
            x[is_txn] = self.txn_proj(txn_features_full[n_id[is_txn]])
        if (~is_txn).any():
            x[~is_txn] = self.value_embedding(n_id[~is_txn] - num_txn)
        return x

    def forward(self, n_id: torch.Tensor, edge_index: torch.Tensor, txn_features_full: torch.Tensor,
                num_txn: int, batch_size: int) -> torch.Tensor:
        x = self.initial_features(n_id, txn_features_full, num_txn)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        seed_embeddings = x[:batch_size]
        logits = self.classifier(seed_embeddings).squeeze(-1)
        return logits
