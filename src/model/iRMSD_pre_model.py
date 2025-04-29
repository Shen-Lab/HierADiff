

import os
import sys
import math

import time


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import BatchNorm
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean, scatter_softmax, scatter_sum, scatter_add

from geom_utils import NoiseSchedule_flexFM_Rt, score_norm
from .utils import _init

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, to_dense_batch
import math

DEBUG_cutoff = False



class AtomEmbedding(nn.Module):
    """
        Using the aa type / sigma noise embedding / LM embedding (optional)
        It is building the node embedding for the protein graph if the input is the graph.x with reduced dimension size


        Embeddings for atom identity only (as of now)

        node_attr with shape (n_res, self.esm_embed_dim=1281 + self.t_embed_dim (+self.msf_embed_dim=32)) -> node_attr with shape (n_res, self.ns)
    """
    def __init__(self, args, num_atom, hidden_dim=32):
        super(AtomEmbedding, self).__init__()
        
        # add 1 for padding
        self.atom_ebd = nn.Embedding(num_atom+1, hidden_dim) # convert integer indices into dense embeddings.

        # print("args.dropout", args.dropout)
        # print("args.layer_norm_eps", args.layer_norm_eps)
        # print("args.lm_embed_dim", args.lm_embed_dim)
        # print("hidden_dim", hidden_dim)
        
        # LM embedding (ESM2)
        self.lm_ebd_dim = args.lm_embed_dim # defaulted as 1280
        if self.lm_ebd_dim > 0:
            self.lm_ebd = nn.Sequential(
                nn.Linear(self.lm_ebd_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim, eps=args.layer_norm_eps),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # _init(self) # initialize the parameters

    def forward(self, x): # x is the node_attr shape (n_res, self.esm_embed_dim=1281 + self.t_embed_dim)
        # atom sequence
        atom_ebd = self.atom_ebd(x[:,self.lm_ebd_dim:self.lm_ebd_dim + 1].long()).squeeze() # x[:,self.lm_ebd_dim:self.lm_ebd_dim + 1] is aa type
        final_ebd = atom_ebd
        # print(final_ebd)
        
        # consider LM embedding here
        if self.lm_ebd_dim > 0:
            x = torch.cat([final_ebd, x[:, :self.lm_ebd_dim]], dim=1)
            # print("x", x)
            final_ebd = self.lm_ebd(x)
            # print("final_ebd", final_ebd)
        return final_ebd


from torch_scatter import scatter_add


class GaussianSmearing(nn.Module):
    # Used to embed the edge distances.
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        # dist: (E, 1)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))  # (E, num_gaussians)


class IEGMNLayer(nn.Module):
    def __init__(self, args, in_dim_h, in_dim_edge, hidden_dim, gaussian_emb_dim):
        super().__init__()
        self.beta = 0.5
        
        # Use GaussianSmearing for distance embedding.
        self.gaussian_smearing = GaussianSmearing(start=0.0, stop=args.rec_max_radius, num_gaussians=args.dist_embed_dim)
        
        # -----------------------
        # Intra-graph modules (with GaussianSmearing embedding)
        # -----------------------
        # Input: [h_i, h_j, edge_attr, GRB(dist)].
        self.intra_edge_mlp = nn.Sequential(
            nn.Linear(2 * in_dim_h + in_dim_edge + args.dist_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # This computes the aggregated intra message m.
        self.intra_pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # -----------------------
        # Cross-graph modules (using only node features with query-key attention)
        # -----------------------
        self.psi_q = nn.Linear(in_dim_h, hidden_dim)  # Query projection.
        self.psi_k = nn.Linear(in_dim_h, hidden_dim)  # Key projection.
        self.W     = nn.Linear(in_dim_h, hidden_dim)  # Value projection.
        
        # -----------------------
        # Node update function φ_h.
        # Input: [current h, aggregated intra message m, aggregated cross message μ, original h (h^(0))].
        # Output dimension: in_dim_h.
        self.phi_h = nn.Sequential(
            nn.Linear(in_dim_h + 2 * hidden_dim + in_dim_h, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim_h)
        )

    def forward(
        self,
        rec_h, rec_pos, rec_edge_index, rec_edge_attr,
        lig_h, lig_pos, lig_edge_index, lig_edge_attr,
        cross_edge_index,
        rec_h0, lig_h0  # original node features for receptor and ligand
    ):
        """
        Inputs:
          - rec_h:  (n_rec, d_h) current receptor node features.
          - rec_pos: (n_rec, 3) receptor coordinates.
          - rec_edge_index: (2, E_rec) intra edges for receptor.
          - rec_edge_attr: (E_rec, d_e) receptor edge features.
          - lig_h, lig_pos, lig_edge_index, lig_edge_attr: analogous for ligand.
          - cross_edge_index: (2, E_cross) where first row are receptor indices,
                              second row are ligand indices (fully connected cross edges).
          - rec_h0, lig_h0: original receptor/ligand node features (h^(0)).
        """
        # --- Intra-graph messages ---
        m_rec, delta_rec = self.intra_message_and_update(rec_h, rec_pos, rec_edge_index, rec_edge_attr)
        m_lig, delta_lig = self.intra_message_and_update(lig_h, lig_pos, lig_edge_index, lig_edge_attr)
        
        # --- Cross-graph messages ---
        mu_rec, _ = self.cross_message_and_update(rec_h, lig_h, cross_edge_index, update_target="receptor")
        mu_lig, _ = self.cross_message_and_update(rec_h, lig_h, cross_edge_index, update_target="ligand")
        
        # --- Residual node update ---
        new_rec_h = (1 - self.beta) * rec_h + self.beta * self.phi_h(torch.cat([rec_h, m_rec, mu_rec, rec_h0], dim=-1))
        new_lig_h = (1 - self.beta) * lig_h + self.beta * self.phi_h(torch.cat([lig_h, m_lig, mu_lig, lig_h0], dim=-1))
        
        # Update positions using only intra messages.
        new_rec_pos = rec_pos + delta_rec
        new_lig_pos = lig_pos + delta_lig
        
        return new_rec_h, new_rec_pos, new_lig_h, new_lig_pos

    
    def intra_message_and_update(self, h, pos, edge_index, edge_attr):
        row, col = edge_index  # shape (2, E)
        diff = pos[row] - pos[col]                      # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)   # (E, 1)
        # Use GaussianSmearing to embed the distance.
        dist_emb = self.gaussian_smearing(dist)         # (E, gaussian_emb_dim)
        # Concatenate: [h_i, h_j, edge_attr, GRB(dist)].
        edge_in = torch.cat([h[row], h[col], edge_attr, dist_emb], dim=-1)
        edge_msg = self.intra_edge_mlp(edge_in)          # (E, hidden_dim)
        # Aggregate intra message for each node.
        agg_msg = scatter_add(edge_msg, row, dim=0, dim_size=h.size(0))
        # Position update: learn scalar weights per edge.
        w_ij = self.intra_pos_mlp(edge_msg)             # (E, 1)
        delta_pos = scatter_add(w_ij * diff, row, dim=0, dim_size=pos.size(0))
        return agg_msg, delta_pos


    def cross_message_and_update(self, rec_h, lig_h, cross_edge_index, update_target="receptor"):
        """
        Computes cross messages from ligand to receptor (or vice versa) using only node features.
        Optimizations applied: 
        - Replaced element-wise multiply & sum with `torch.bmm`
        - Used `scatter_softmax` for CUDA optimization
        - Avoided multiple `scatter_add` calls for efficiency
        """
        rec_idx, lig_idx = cross_edge_index

        if update_target == "receptor":
            target_features = rec_h[rec_idx]  # Receptor nodes
            source_features = lig_h[lig_idx]  # Ligand nodes
            node_count = rec_h.size(0)
            idx_for_scatter = rec_idx
        else:
            target_features = lig_h[lig_idx]  # Ligand nodes
            source_features = rec_h[rec_idx]  # Receptor nodes
            node_count = lig_h.size(0)
            idx_for_scatter = lig_idx

        # Compute query, key, and value projections
        query = self.psi_q(target_features)  # (E_cross, hidden_dim)
        key = self.psi_k(source_features)    # (E_cross, hidden_dim)
        value = self.W(source_features)      # (E_cross, hidden_dim)

        # Optimized dot product attention computation
        att_logits = torch.bmm(query.unsqueeze(1), key.unsqueeze(2)).squeeze(-1)  # (E_cross, 1)

        # Use optimized `scatter_softmax`
        att_weight = scatter_softmax(att_logits, idx_for_scatter, dim=0)  # (E_cross, 1)

        # Compute attention-weighted message
        weighted_msg = att_weight * value  # (E_cross, hidden_dim)

        # Efficient scatter sum
        agg_msg = scatter_add(weighted_msg, idx_for_scatter, dim=0, dim_size=node_count)

        # No coordinate update for cross edges
        delta_pos = torch.zeros((node_count, 3), device=agg_msg.device)

        return agg_msg, delta_pos


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add, scatter_softmax

# ---------------------------
# Global Attention Pooling Module
# ---------------------------
class GlobalAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super(GlobalAttentionPool, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, batch):
        # x: (N, in_dim), batch: (N,)
        scores = self.att_mlp(x)        # (N, 1)
        scores = scores.squeeze(-1)      # (N,)
        att_weights = scatter_softmax(scores, batch)  # (N,)
        att_weights = att_weights.unsqueeze(-1)       # (N, 1)
        out = scatter_add(att_weights * x, batch, dim=0)
        return out

# ---------------------------
# iRMSD Predictor with Attention-Weighted Pooling
# ---------------------------
class IRMSDPredictorAttention(nn.Module):
    def __init__(self, node_feat_dim, pool_hidden_dim=32, mlp_hidden_dim=64):
        super(IRMSDPredictorAttention, self).__init__()
        # Two separate attention pooling modules for receptor and ligand
        self.rec_pool = GlobalAttentionPool(node_feat_dim, hidden_dim=pool_hidden_dim)
        self.lig_pool = GlobalAttentionPool(node_feat_dim, hidden_dim=pool_hidden_dim)
        # MLP head to predict a scalar iRMSD from concatenated pooled features.
        in_dim = 2 * node_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1)
        )
        # Lower and upper bounds for iRMSD.
        self.lower_bound = 0.35
        self.upper_bound = 8.0
        
    def forward(self, rec_h, rec_batch, lig_h, lig_batch):
        # Compute global attention-based pooling for receptor and ligand.
        rec_global = self.rec_pool(rec_h, rec_batch)  # shape: (num_graphs, node_feat_dim)
        lig_global = self.lig_pool(lig_h, lig_batch)    # shape: (num_graphs, node_feat_dim)
        # Concatenate pooled features.
        global_feat = torch.cat([rec_global, lig_global], dim=-1)  # shape: (num_graphs, 2*node_feat_dim)
        # Get raw prediction.
        raw_pred = self.mlp(global_feat)  # shape: (num_graphs, 1)
        # Restrict prediction to (0.35, 8.0) using a sigmoid scaling.
        irmsd_pred = self.lower_bound + (self.upper_bound - self.lower_bound) * ((torch.tanh(raw_pred) + 1) / 2)
        return irmsd_pred


class InterfacePredictor(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32):
        super(InterfacePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification output
        )

    def forward(self, node_features):
        logits = self.mlp(node_features).squeeze(-1)  # Shape: (num_nodes,)
        return logits  # Will apply sigmoid outside for probabilities







# !!!!!!!
def from_dense_batch(x_dense, mask):
    """
    Args:
        x_dense (Tensor): Padded tensor of shape [B, L, F]
        mask (Tensor): Boolean mask of shape [B, L] indicating valid positions.
    Returns:
        x_flat (Tensor): Flattened tensor of shape [N, F] where N is total valid nodes.
        mask_flat (Tensor): Flattened mask of shape [N]
    """
    x_flat = x_dense[mask]  # Select only the valid nodes.
    mask_flat = mask.view(-1)
    return x_flat, mask_flat

# === EGNN Layer (Equivariant Updates) ===
class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EGNNLayer, self).__init__(aggr='add')
        # We expect the input to the MLP to be a concatenation of features from two nodes.
        self.fc = nn.Linear(in_channels * 2, out_channels)
        # For coordinate updates, we use a linear layer on the same concatenated features.
        self.coord_mlp = nn.Linear(in_channels * 2, 1)

    def forward(self, x, edge_index, pos):
        row, col = edge_index  # row: target indices, col: source indices for each edge
        
        # For each edge, concatenate the features of the target and source nodes.
        edge_features = torch.cat([x[row], x[col]], dim=-1)  # shape: (n_edges, 2 * in_channels)
        
        # Compute edge-level message updates.
        edge_updates = self.fc(edge_features)  # shape: (n_edges, out_channels)
        # Aggregate these edge updates for each target node.
        node_updates = scatter_add(edge_updates, row, dim=0, dim_size=x.size(0))
        # Update node features.
        x = x + node_updates

        # --- Coordinate Update ---
        # Compute a scalar weight per edge.
        coord_updates = self.coord_mlp(edge_features)  # shape: (n_edges, 1)
        # Multiply the difference in positions (target - source) by these weights.
        delta_pos_edges = (pos[row] - pos[col]) * coord_updates  # shape: (n_edges, 3)
        # Aggregate these coordinate updates per node.
        delta_pos = scatter_add(delta_pos_edges, row, dim=0, dim_size=pos.size(0))
        # Update positions.
        pos = pos + delta_pos

        return x, pos

# === EGNN Encoder ===
class EGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super(EGNNEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EGNNLayer(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, pos):
        for layer in self.layers:
            x, pos = layer(x, edge_index, pos)
        return x, pos


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x_source, batch_source, x_target, batch_target):
        """
        Compute cross-attention in a batched manner.
        
        Args:
            x_source (Tensor): Source node features (queries) [N_source, hidden_dim].
            batch_source (Tensor): Batch indices for source nodes [N_source].
            x_target (Tensor): Target node features (keys/values) [N_target, hidden_dim].
            batch_target (Tensor): Batch indices for target nodes [N_target].
        
        Returns:
            output (Tensor): Cross-attention output for source nodes [N_source, hidden_dim].
        """
        # Convert node features to dense batches.
        x_source_dense, mask_source = to_dense_batch(x_source, batch_source)  # [B, L_source, hidden_dim]
        x_target_dense, mask_target = to_dense_batch(x_target, batch_target)  # [B, L_target, hidden_dim]
        
        # Linear projections.
        q = self.q_proj(x_source_dense)  # [B, L_source, hidden_dim]
        k = self.k_proj(x_target_dense)  # [B, L_target, hidden_dim]
        v = self.v_proj(x_target_dense)  # [B, L_target, hidden_dim]
        
        # Compute attention scores.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, L_source, L_target]
        # Mask padded positions in target.
        attn_scores = attn_scores.masked_fill(~mask_target.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L_source, L_target]
        
        # Compute weighted sum over values.
        output_dense = torch.matmul(attn_weights, v)  # [B, L_source, hidden_dim]
        # Convert dense batch back to the original (flattened) shape.
        output, _ = from_dense_batch(output_dense, mask_source)  # [N_source, hidden_dim]
        return output



# !!!!!!!

# ---------------------------
# (The rest of your modules remain unchanged)
# ---------------------------
# (AtomEmbedding, GaussianSmearing, IEGMNLayer)
# For brevity, assume they are defined as in your provided code.

# ---------------------------
# Modified IEGMN Model with Attention-based iRMSD Prediction
# ---------------------------
class IEGMN(nn.Module):
    def __init__(self, args, model_params, node_in_dim=32, edge_in_dim=32, hidden_dim=32,
                 num_layers=1, gaussian_emb_dim=32):
        super().__init__()

        self.args = args

        self.intra_max_radius = args.intra_max_radius
        num_atoms = model_params["num_residues"]
        self.atom_embed = AtomEmbedding(args, num_atoms, hidden_dim=hidden_dim)
        self.rec_dist_exp = GaussianSmearing(start=0.0, stop=args.rec_max_radius, num_gaussians=hidden_dim)

        if True:
            self.layers = nn.ModuleList([
                IEGMNLayer(
                    args,
                    in_dim_h=node_in_dim,
                    in_dim_edge=edge_in_dim,
                    hidden_dim=hidden_dim,
                    gaussian_emb_dim=gaussian_emb_dim
                ) for _ in range(num_layers)
            ])

            # Replace the previous iRMSD predictor with our attention-weighted version.
            self.irmsd_predictor = IRMSDPredictorAttention(node_feat_dim=node_in_dim, 
                                                            pool_hidden_dim=32, mlp_hidden_dim=hidden_dim)
            if args.interface_reg:
                # Interface predictor (new auxiliary loss)
                self.interface_predictor = InterfacePredictor(node_feat_dim=node_in_dim, hidden_dim=hidden_dim)

        else:
            self.egnn = EGNNEncoder(node_in_dim, hidden_dim)
            self.cross_attention = CrossAttention(hidden_dim)

            self.interface_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

            # Regression head for iRMSD. We use the same head on combined features.
            self.irmsd_regressor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

            # Final regressor that uses aggregated iRMSD predictions plus graph sizes.
            self.final_regressor = nn.Sequential(
                nn.Linear(2, 2),
                nn.ReLU(),
                nn.Linear(2, 1)
            )


            

        

    def forward(self, batch):
        # N irmsd correlation

        # Build radius graphs for ligand and receptor.
        node_attr_lig, edge_index_lig, edge_attr_lig = self.build_radius_graph(batch, "ligand")
        node_attr_rec, edge_index_rec, edge_attr_rec = self.build_radius_graph(batch, "receptor")

        # print("node_attr_lig", node_attr_lig)
        # print("node_attr_rec", node_attr_rec)
        
        rec = batch['receptor']
        lig = batch['ligand']

        rec_h0 = node_attr_rec
        lig_h0 = node_attr_lig

        rec_h = node_attr_rec
        rec_pos = rec.pos
        rec_edge_index = edge_index_rec
        rec_edge_attr = edge_attr_rec

        lig_h = node_attr_lig
        lig_pos = lig.pos
        lig_edge_index = edge_index_lig
        lig_edge_attr = edge_attr_lig

        # Compute fully-connected cross edges for the entire batch.
        n_rec = rec_h.size(0)
        n_lig = lig_h.size(0)
        rec_idx = torch.arange(n_rec, device=rec_h.device).unsqueeze(1).repeat(1, n_lig).view(-1)
        lig_idx = torch.arange(n_lig, device=lig_h.device).unsqueeze(0).repeat(n_rec, 1).view(-1)
        cross_edge_index = torch.stack([rec_idx, lig_idx], dim=0)

        if True:
            for layer in self.layers:
                rec_h, rec_pos, lig_h, lig_pos = layer(
                    rec_h, rec_pos, rec_edge_index, rec_edge_attr,
                    lig_h, lig_pos, lig_edge_index, lig_edge_attr,
                    cross_edge_index,
                    rec_h0, lig_h0
                )
            
            # Use the attention-based iRMSD predictor.
            irmsd_pred = self.irmsd_predictor(rec_h, rec.batch, lig_h, lig.batch)  # (num_graphs, 1)

            

            if self.args.interface_reg:
                # Compute interface residue logits
                interface_logits_rec = self.interface_predictor(rec_h)  # (num_rec_nodes,)
                interface_logits_lig = self.interface_predictor(lig_h)  # (num_lig_nodes,)

                # Convert logits to probabilities
                interface_pred_rec = torch.sigmoid(interface_logits_rec)  # (num_rec_nodes,)
                interface_pred_lig = torch.sigmoid(interface_logits_lig)  # (num_lig_nodes,)

                return {
                    'receptor': {'x': rec_h, 'pos': rec_pos, 'batch': rec.batch, 'interface_pred': interface_pred_rec},
                    'ligand': {'x': lig_h, 'pos': lig_pos, 'batch': lig.batch, 'interface_pred': interface_pred_lig},
                    'irmsd_pred': irmsd_pred
                }
            else:
                interface_pred_rec = None
                interface_pred_lig = None

                return {
                    'receptor': {'x': rec_h, 'pos': rec_pos, 'batch': rec.batch, 'interface_pred': interface_pred_rec},
                    'ligand': {'x': lig_h, 'pos': lig_pos, 'batch': lig.batch, 'interface_pred': interface_pred_lig},
                    'irmsd_pred': irmsd_pred
                }
            


        else:
            x1 = node_attr_lig          # [N1_total, in_channels]
            pos1 = lig_pos      # [N1_total, 3]
            edge_index1 = edge_index_lig
            batch1 = lig.batch  # [N1_total]
            
            # Extract protein_2 data.
            x2 = node_attr_rec          # [N2_total, in_channels]
            pos2 = rec_pos      # [N2_total, 3]
            edge_index2 = edge_index_rec
            batch2 = rec.batch  # [N2_total]

            # print("x1", x1)
            # # print("cross_attn_out_12", cross_attn_out_12)
            # print("x2", x2)
            # # print("cross_attn_out_21", cross_attn_out_21)


            # Encode proteins using EGNN.
            x1, pos1 = self.egnn(x1, edge_index1, pos1)
            x2, pos2 = self.egnn(x2, edge_index2, pos2)
            
            # Compute cross-attention outputs in both directions.
            # Protein_1 nodes attend to protein_2 nodes.
            cross_attn_out_12 = self.cross_attention(x1, batch1, x2, batch2)
            # Protein_2 nodes attend to protein_1 nodes.
            cross_attn_out_21 = self.cross_attention(x2, batch2, x1, batch1)
            
            # Use the combination (e.g., sum) of original and attended features for interface prediction.
            interface_logits_1 = self.interface_classifier(x1 + cross_attn_out_12).squeeze(-1)
            interface_logits_2 = self.interface_classifier(x2 + cross_attn_out_21).squeeze(-1)
            interface_pred_1 = torch.sigmoid(interface_logits_1)
            interface_pred_2 = torch.sigmoid(interface_logits_2)

            
            combined_feat_1 = torch.cat([x1, cross_attn_out_12], dim=-1)  # [N1_total, 2*hidden_dim]
            combined_feat_2 = torch.cat([x2, cross_attn_out_21], dim=-1)  # [N2_total, 2*hidden_dim]

            
            
            node_rmsd_pred_1 = self.irmsd_regressor(combined_feat_1)  # [N1_total, 1]
            node_rmsd_pred_2 = self.irmsd_regressor(combined_feat_2)  # [N2_total, 1]

            


            
            # Aggregate node predictions to form a per-sample prediction.
            rmsd_pred_1 = scatter_mean(node_rmsd_pred_1, batch1, dim=0)  # [B, 1]
            rmsd_pred_2 = scatter_mean(node_rmsd_pred_2, batch2, dim=0)  # [B, 1]


            # Additionally, compute the graph sizes (number of nodes) for each protein.
            count1 = scatter_sum(torch.ones(x1.size(0), device=x1.device), batch1, dim=0)  # [B]
            count2 = scatter_sum(torch.ones(x2.size(0), device=x2.device), batch2, dim=0)  # [B]

            # # Concatenate the two per-sample RMSD predictions and the graph sizes.
            # combined_regressor_input = torch.cat([rmsd_pred_1, rmsd_pred_2,
            #                                         count1.unsqueeze(-1).float(),
            #                                         count2.unsqueeze(-1).float()], dim=-1)  # [B, 4]

            combined_regressor_input = torch.cat([rmsd_pred_1, rmsd_pred_2], dim=-1)  # [B, 4]
                                                    
                
                
            # Final iRMSD prediction.
            rmsd_pred = self.final_regressor(combined_regressor_input).squeeze(-1)  # [B]

            # constraint the iRMSD to be in the range of possible values
            lower_bound = 0.35
            upper_bound = 8.0
            rmsd_pred = lower_bound + (upper_bound - lower_bound) * ((torch.tanh(rmsd_pred) + 1) / 2)
        
            return {
                'receptor': {'interface_pred': interface_pred_2, 'batch': rec.batch},
                'ligand': {'interface_pred': interface_pred_1, 'batch': lig.batch},
                'irmsd_pred': rmsd_pred
            }



    def build_radius_graph(self, batch, key):
        """
            proteins radius graph
            Adds noise information to existing embeddings.
            @return node_attr, edge_index, edge_attr, edge_sh
        """
        DEBUG = False
        
        # if no ESM models, graph.x should still be flat
        if len(batch[key].x.shape) == 1:
            batch[key].x = batch[key].x[:,None]
        
        # node_attr.shape == (n_res, self.esm_embed_dim=1281)
        node_attr = batch[key].x

        # print("node_attr", node_attr)

        node_attr = self.atom_embed(node_attr)

        # print("node_attr after", node_attr)

        radius_edges = radius_graph(batch[key].pos, self.intra_max_radius, batch[key].batch) #### [?]
    
        # batch[key, key].edge_index should be (2, n_res)
        edge_index = radius_edges.long()
        
        # # this assumes the edges were already created in preprocessing since protein's structure is fixed
        # edge_index = batch[key, key].edge_index
        src, dst = edge_index
        edge_vec = batch[key].pos[dst.long()] - batch[key].pos[src.long()] # [n_edge, 3]
        if DEBUG:
            print('edge_vec.shape', edge_vec.shape)

        edge_attr = self.rec_dist_exp(edge_vec.norm(dim=-1)) # [n_edge, n_gauss=32]
        
        
        # node_attr.shape: (n_res, hidden_dim)
        # edge_index.shape: (2, n_res)
        # edge_attr.shape: (n_edge, hidden_dim)
        
        return node_attr, edge_index, edge_attr



##############################################################################
# Usage example:
#
# model = IEGMN(node_in_dim=32, edge_in_dim=32, hidden_dim=64,
#              num_layers=3, gaussian_emb_dim=50)
# out = model(batch)
# final_rec_x, final_rec_pos = out['receptor']['x'], out['receptor']['pos']
# final_lig_x, final_lig_pos = out['ligand']['x'], out['ligand']['pos']
##############################################################################








