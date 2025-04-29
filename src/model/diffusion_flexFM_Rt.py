

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
from torch_scatter import scatter, scatter_mean, scatter_softmax, scatter_sum

from geom_utils import NoiseSchedule_flexFM_Rt, score_norm
from .utils import _init

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

DEBUG_cutoff = False






class CrossAttentionLayer(nn.Module):
    """ Cross-attention layer for message passing between two graphs. output shape: (n_edge, out_dim)
    """
    def __init__(self, in_dim, out_dim, num_heads=4, use_edge_bias=False, edge_dim=3):
        """
        in_dim: input feature dimension (for receptor and ligand)
        out_dim: output dimension (should be divisible by num_heads)
        num_heads: number of attention heads
        use_edge_bias: whether to incorporate edge positional information
        edge_dim: dimension of the positional vector (e.g., 3 for 3D)
        """
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads

        # Linear projections for queries (receptor) and keys/values (ligand)
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        # Optional: incorporate edge features (e.g., relative positional vector)
        self.use_edge_bias = use_edge_bias
        if use_edge_bias:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, num_heads)  # one bias per head
            )
        
        # Final projection to mix heads back
        self.fc_out = nn.Linear(out_dim, out_dim)

    def forward(self, receptor_features, ligand_features, edge_index, edge_vec=None):
        """
        receptor_features: Tensor of shape (n_rec, in_dim) (dst nodes, query)
        ligand_features: Tensor of shape (n_lig, in_dim) (src nodes)
        edge_index: Tensor of shape (2, n_edge), where edge_index[0] = src (ligand indices),
                    and edge_index[1] = dst (receptor indices).
        edge_vec: (optional) Tensor of shape (n_edge, edge_dim) containing edge geometric info.
        """
        src, dst = edge_index  # ligand indices (src) and receptor indices (dst)

        # Compute projections
        Q = self.W_q(receptor_features)  # (n_rec, out_dim)
        K = self.W_k(ligand_features)      # (n_lig, out_dim)
        V = self.W_v(ligand_features)      # (n_lig, out_dim)

        # Select features for the edges
        Q_edges = Q[dst]   # (n_edge, out_dim)
        K_edges = K[src]   # (n_edge, out_dim)
        V_edges = V[src]   # (n_edge, out_dim)

        # Reshape for multi-head attention: (n_edge, num_heads, d_k)
        Q_edges = Q_edges.view(-1, self.num_heads, self.d_k)
        K_edges = K_edges.view(-1, self.num_heads, self.d_k)
        V_edges = V_edges.view(-1, self.num_heads, self.d_k)

        # Compute scaled dot-product attention score for each head
        # Here we compute a dot product for each head and scale by sqrt(d_k)
        attn_scores = (Q_edges * K_edges).sum(dim=-1) / math.sqrt(self.d_k)  # (n_edge, num_heads)

        # Optionally, add a bias from the edge (e.g., edge_vec)
        if self.use_edge_bias and edge_vec is not None:
            # Project edge_vec to get a bias per head for each edge.
            edge_bias = self.edge_mlp(edge_vec)  # (n_edge, num_heads)
            attn_scores = attn_scores + edge_bias

        # Normalize attention scores for each receptor node.
        # Note: we group edges by their receptor node index (dst) using scatter_softmax.
        attn_weights = scatter_softmax(attn_scores, dst, dim=0)  # (n_edge, num_heads)

        # Weight the ligand value features by the computed attention weights.
        weighted_values = V_edges * attn_weights.unsqueeze(-1)  # (n_edge, num_heads, d_k)

        # # Aggregate the weighted values for each receptor node.
        # # We sum over all ligand neighbors for each receptor.
        # aggregated = scatter_sum(weighted_values, dst, dim=0, dim_size=receptor_features.size(0))
        # # aggregated shape: (n_rec, num_heads, d_k)

        # # Reshape back to (n_rec, out_dim) and combine heads.
        # aggregated = aggregated.view(receptor_features.size(0), -1)

        out = weighted_values.view(weighted_values.size(0), -1)
        out = self.fc_out(out)

        return out

# Example usage:
# Assume:
#   receptor_feats: Tensor of shape (n_rec, 32)
#   ligand_feats:   Tensor of shape (n_lig, 32)
#   edge_index:     Tensor of shape (2, n_edge)
#   edge_vec:       Tensor of shape (n_edge, 3)
#
# cross_attn = CrossAttentionLayer(in_dim=32, out_dim=64, num_heads=4, use_edge_bias=True)
# enriched_features = cross_attn(receptor_feats, ligand_feats, edge_index, edge_vec)



class AtomEmbedding(nn.Module):
    """
        Using the aa type / sigma noise embedding / LM embedding (optional)
        It is building the node embedding for the protein graph if the input is the graph.x with reduced dimension size


        Embeddings for atom identity only (as of now)

        node_attr with shape (n_res, self.esm_embed_dim=1281 + self.t_embed_dim (+self.msf_embed_dim=32)) -> node_attr with shape (n_res, self.ns)
    """
    def __init__(self, args, num_atom, no_time=False, use_nm_msf_feat=False):
        super(AtomEmbedding, self).__init__()
        self.use_nm_msf_feat = use_nm_msf_feat
        self.no_time = no_time
        self.sigma_embed_dim = args.sigma_embed_dim
        # add 1 for padding
        self.atom_ebd = nn.Embedding(num_atom+1, args.ns) # convert integer indices into dense embeddings.
        self.sigma_ebd = nn.Linear(args.sigma_embed_dim, args.ns)
        # LM embedding (ESM2)
        self.lm_ebd_dim = args.lm_embed_dim # defaulted as 1280
        if self.lm_ebd_dim > 0:

            if not args.layer_norm:
                if self.use_nm_msf_feat:
                    self.lm_ebd = nn.Linear(self.lm_ebd_dim + args.ns + args.msf_embed_dim, args.ns)
                else:
                    self.lm_ebd = nn.Linear(self.lm_ebd_dim + args.ns, args.ns)
            else:
                if self.use_nm_msf_feat:
                    self.lm_ebd = nn.Sequential(
                        nn.Linear(self.lm_ebd_dim + args.ns + args.msf_embed_dim, args.ns),
                        nn.ReLU(),
                        nn.LayerNorm(args.ns, eps=args.layer_norm_eps),
                        nn.Dropout(args.dropout),
                        nn.Linear(args.ns, args.ns),
                    )
                else:
                    self.lm_ebd = nn.Sequential(
                        nn.Linear(self.lm_ebd_dim + args.ns, args.ns),
                        nn.ReLU(),
                        nn.LayerNorm(args.ns, eps=args.layer_norm_eps),
                        nn.Dropout(args.dropout),
                        nn.Linear(args.ns, args.ns),
                    )

        _init(self) # initialize the parameters

    def forward(self, x): # x is the node_attr shape (n_res, self.esm_embed_dim=1281 + self.t_embed_dim)
        # atom sequence
        atom_ebd = self.atom_ebd(x[:,self.lm_ebd_dim:self.lm_ebd_dim + 1].long()).squeeze() # x[:,self.lm_ebd_dim:self.lm_ebd_dim + 1] is aa type
        # sigma noise embedding
        sigma_start = 1 + self.lm_ebd_dim # dim for aa type + dim for lm embedding
        sigma_end = sigma_start + self.sigma_embed_dim

        if self.no_time:
            final_ebd = atom_ebd
        else:
            sigma_ebd = self.sigma_ebd(x[:,sigma_start:sigma_end])
            # add together
            final_ebd = atom_ebd + sigma_ebd # [n_res, self.ns]
        
        # consider LM embedding here
        if self.lm_ebd_dim > 0:

            if self.use_nm_msf_feat:
                x = torch.cat([final_ebd, x[:, :self.lm_ebd_dim], x[:, sigma_end:]], dim=1)
                final_ebd = self.lm_ebd(x)
            else:
                x = torch.cat([final_ebd, x[:, :self.lm_ebd_dim]], dim=1)
                final_ebd = self.lm_ebd(x)
        return final_ebd



# [TODO: weights in the sh is only the edge attr]
class TPCL(nn.Module):
    """
        Tensor product convolution layer
    """
    def __init__(self, args,
                 in_irreps, sh_irreps, out_irreps, n_edge_features,
                 residual=True, hidden_features=None, is_last_layer=False, is_layerNorm_edgeFC=False):
        super(TPCL, self).__init__()
        self.in_irreps = in_irreps # input seq of irreps
        self.out_irreps = out_irreps # output seq of irreps
        self.sh_irreps = sh_irreps # the sh module up to l=2
        self.residual = residual # boolean value
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )
        if not is_layerNorm_edgeFC:
            self.fc = nn.Sequential(
                nn.Linear(n_edge_features, hidden_features),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_features, self.tensor_prod.weight_numel),
            )
        else: # for fusion of the distance length and attention edge features
            self.fc = nn.Sequential(
                nn.Linear(n_edge_features, hidden_features),
                nn.ReLU(),
                nn.LayerNorm(hidden_features),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_features, self.tensor_prod.weight_numel),
            )
        if not args.no_batch_norm and not is_last_layer: # args.no_batch_norm defaulted as False
            # if not last layer 
            self.batch_norm = BatchNorm(out_irreps)
        else:
            # if last layer
            self.batch_norm = None

        _init(self)

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        NOTE: out_nodes is the number of nodes in the INPUT graph that is represented with "edge_index" variable 
                            OR leave it as None to use the number of nodes in the graph that is represented with "node_attr" variable
                            (DO NOT use any other values; All above is according to the torch_scatter.scatter function below)


        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        @return out [N, out_irreps]
        """
        DEBUG = False
        edge_src, edge_dst = edge_index

        if DEBUG:
            print('edge_index.shape', edge_index.shape)


        if DEBUG:
            print('node_attr.shape', node_attr.shape)
            print('edge_attr.shape', edge_attr.shape)
            print('edge_sh.shape', edge_sh.shape)
            print('self.fc(edge_attr).shape', self.fc(edge_attr).shape)

        #
        # node_attr[edge_dst] shape (n_edge, self.ns) [inv]
        # edge_sh shape (n_edge, 9) [equi]
        # self.fc(edge_attr) shape (n_edge, self.tensor_prod.weight_numel) [inv]
        # forward(x, y[, weight])
        # tensor product bt dst_node_attr and edge_sh using edge_attr as weight
        # tp shape (n_edge, self.out_irreps_size)
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        if DEBUG:
            print('tp.shape', tp.shape) # (n_edge, out_irreps_size (e.g. f"{ns}x0e + {nv}x1o" as out_irreps, then out_irreps_size as ns + nv*3))

        if DEBUG:
            print('before_out_nodes', out_nodes) # None
        out_nodes = out_nodes or node_attr.shape[0] # node_attr is either specified as out_nodes or # of nodes in the graph
        if DEBUG:
            print('out_nodes', out_nodes) # n_res

        #
        # aggregate the tensor tp based on the indices in edge_src, along the first dimension (rows) of tp. The aggregation is performed per index, and the type of aggregation is determined by the reduction parameter. The result of this aggregation is stored in the tensor out.
        try:
            out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)
        except:
            print('!!!!!!!!!!!!!! error in scatter !!!!!!!!!!!!!')
            print('tp.shape', tp.shape)
            print('max edge_src', edge_src.max())
            print('edge_src.shape', edge_src.shape)
            print('out_nodes', out_nodes)
            print('reduction', reduction)
        # aggregating the data in the tensor tp based on the indices in edge_src, along the first dimension (rows) of tp. The aggregation is performed per index, and the type of aggregation is determined by the reduction parameter. The result of this aggregation is stored in the tensor out.

        if DEBUG:
            print('out.shape', out.shape) # (n_res, out_irreps_size (e.g. f"{ns}x0e + {nv}x1o" as out_irreps, then out_irreps_size as ns + nv*3))

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        if DEBUG:
            print('out_normed.shape', out.shape) # (n_res, out_irreps_size (e.g. f"{ns}x0e + {nv}x1o" as out_irreps, then out_irreps_size as ns + nv*3))

        return out

class flexFM_RtScoreModel(torch.nn.Module):
    def __init__(self, args, model_params,
        confidence_mode=False,
        confidence_dropout=0,
        confidence_no_batchnorm=True,
        num_confidence_outputs=1,
    ):
        super(flexFM_RtScoreModel, self).__init__()
        self.noise_schedule = NoiseSchedule_flexFM_Rt(args)
        self.t_embedding = get_timestep_embedding(args)

        self.in_lig_edge_features = 4
        self.max_radius = args.max_radius
        self.max_radius_bond = args.max_radius # defaulted as 5.
        self.intra_max_radius = args.intra_max_radius # defaulted as 10.
        self.rec_max_radius = args.rec_max_radius  # defaulted as 30. # this is used in gaussian smearing only (not in radius_graph)
        self.cross_max_dist = args.cross_max_dist # defaulted as 80
        self.dynamic_max_cross = args.dynamic_max_cross
        self.cross_cutoff_weight = args.cross_cutoff_weight
        self.cross_cutoff_bias = args.cross_cutoff_bias
        self.center_max_dist = 30 # this is used in gaussian smearing
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2) # the sh up to l=2
        self.ns, self.nv = args.ns, args.nv
        ns, nv = self.ns, self.nv  # >>> sigh stupid notation lazy
        self.scale_by_sigma = args.scale_by_sigma

        self.msf_embed_dim = args.msf_embed_dim
        self.nm_corr_embed_dim = args.nm_corr_embed_dim

        #
        self.no_nm = args.no_nm
        self.confidence_mode = confidence_mode
        self.num_conv = args.num_conv_layers # defaulted as 4

        #
        self.max_num_cross_neighbors = args.max_num_cross_neighbors # defaulted as 10

        #
        self.args = args

        #
        self.arcsinh_norm = args.arcsinh_norm
        self.tanh_norm = args.tanh_norm

        num_atoms = model_params["num_residues"]
        self.atom_embed = AtomEmbedding(args, num_atoms, use_nm_msf_feat=args.use_nm_msf_feat)

        # protein edge encoder
        # use rec_edge_embed for both rec/lig
        if not args.layer_norm:
            if args.use_nm_mode_correlation_feat:
                self.rec_edge_embed = nn.Sequential(
                    nn.Linear(args.sigma_embed_dim + args.dist_embed_dim + args.nm_corr_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, ns),
                )
            else:
                self.rec_edge_embed = nn.Sequential(
                    nn.Linear(args.sigma_embed_dim + args.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, ns),
                ) # (n_res, args.sigma_embed_dim + args.dist_embed_dim) -> (n_res, ns)

            self.cross_edge_embed = nn.Sequential(
                nn.Linear(args.sigma_embed_dim + args.cross_dist_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(ns, ns),
            ) # (n_res, args.sigma_embed_dim + args.cross_dist_embed_dim) -> (n_res, ns)
        else:
            if args.use_nm_mode_correlation_feat:
                self.rec_edge_embed = nn.Sequential(
                    nn.Linear(args.sigma_embed_dim + args.dist_embed_dim + args.nm_corr_embed_dim, ns),
                    nn.ReLU(),
                    nn.LayerNorm(ns, eps=args.layer_norm_eps),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, ns),
                )
            else:
                self.rec_edge_embed = nn.Sequential(
                    nn.Linear(args.sigma_embed_dim + args.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.LayerNorm(ns, eps=args.layer_norm_eps),
                    nn.Dropout(args.dropout),
                    nn.Linear(ns, ns),
                )

            self.cross_edge_embed = nn.Sequential(
                nn.Linear(args.sigma_embed_dim + args.cross_dist_embed_dim, ns),
                nn.ReLU(),
                nn.LayerNorm(ns, eps=args.layer_norm_eps),
                nn.Dropout(args.dropout),
                nn.Linear(ns, ns),
            )
        if args.use_nm_msf_feat:
            self.nm_msf_grb_embed = GaussianSmearing(
                -3.0, 5, args.msf_embed_dim
            )
        
        if args.use_nm_mode_correlation_feat:
            self.nm_corr_grb_embed = GaussianSmearing(
                -1.0, 1, args.nm_corr_embed_dim
            )

        # distance expansion
        # GaussianSmearing(self, start, stop, num_gaussians)
        # (0.0, 30, 32)
        self.lig_dist_exp = GaussianSmearing(
                0.0, self.rec_max_radius, args.dist_embed_dim) # self.intra_max_radius defaulted as 10
        # (0.0, 30, 32)
        self.rec_dist_exp = GaussianSmearing(
                0.0, self.rec_max_radius, args.dist_embed_dim)
        # (0.0, 80, 32)
        self.cross_dist_exp = GaussianSmearing(
                0.0, self.cross_max_dist, args.cross_dist_embed_dim) # self.cross_max_dist defaulted as 80 # args.cross_dist_embed_dim defaulted as 32

        if args.use_second_order_repr: # defaulted as False
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]

        intra_convs = []
        cross_convs = []
        lig_intra_convs = []
        lig_cross_convs = []
        interModuleInterAttentions = []
        for i in range(self.num_conv): 
            # self.num_conv defaulted as 4: (in_irreps's ind, out_irreps's ind): (0, 1) -> (1, 2) -> (2, 3) -> (3, 3)
            # if self.num_conv == 5: (in_irreps's ind, out_irreps's ind): (0, 1) -> (1, 2) -> (2, 3) -> (3, 3) -> (3, 3)
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]

            if args.res_rot_flex and i == 0:
                in_irreps = f"{in_irreps} + 2x1o"
            else:
                in_irreps = in_irreps

            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            params_intra = {
                "in_irreps": in_irreps, # input seq of irreps
                "sh_irreps": self.sh_irreps, # the sh up to l=2
                "out_irreps": out_irreps, # output seq of irreps
                "n_edge_features": 3 * ns, # number of edge features
                "hidden_features": 3 * ns, # number of hidden features
                "residual": False, # defaulted as False
            }

            params_inter = {
                "in_irreps": in_irreps, # input seq of irreps
                "sh_irreps": self.sh_irreps, # the sh up to l=2
                "out_irreps": out_irreps, # output seq of irreps
                "n_edge_features": 3 * ns, # number of edge features
                "hidden_features": 3 * ns, # number of hidden features
                "residual": False, # defaulted as False
                "is_layerNorm_edgeFC": args.allInterAttentionFeature, # defaulted as False
            }

            if args.interModuleNoSharedWeight:
                lig_intra_convs.append(TPCL(args, **params_intra))
                lig_cross_convs.append(TPCL(args, **params_inter))

            intra_convs.append(TPCL(args, **params_intra))
            cross_convs.append(TPCL(args, **params_inter))

            if args.allInterAttentionFeature:
                interModuleInterAttentions.append(CrossAttentionLayer(
                    in_dim=ns, out_dim=2*ns, num_heads=4, use_edge_bias=False, edge_dim=3
                )) # [TODO: no time/distance info]

        if args.interModuleNoSharedWeight:
            self.lig_intra_convs = nn.ModuleList(lig_intra_convs)
            self.lig_cross_convs = nn.ModuleList(lig_cross_convs)

        self.intra_convs = nn.ModuleList(intra_convs)
        self.cross_convs = nn.ModuleList(cross_convs)

        if args.allInterAttentionFeature:
            self.interModuleInterAttentions = nn.ModuleList(interModuleInterAttentions)

        if args.indReadoutFlex_sharedWeight or args.indReadoutFlex_noSharedWeight:
            if args.readOutInterAttentionFeature or args.allInterAttentionFeature:
                self.readOutInterAttention = CrossAttentionLayer(
                    in_dim=ns, out_dim=2*ns, num_heads=4, use_edge_bias=False, edge_dim=3
                )

            # output rec coords
            self.last_intra_conv_rec = TPCL(
                args,
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=irrep_seq[-1],
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=False,
            ) 

            self.last_inter_conv_rec = TPCL(
                args,
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=irrep_seq[-1],
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=False,
                is_last_layer=True,
                is_layerNorm_edgeFC=args.readOutInterAttentionFeature or args.allInterAttentionFeature,
            ) 

            if args.indReadoutFlex_noSharedWeight:
                # output lig coords
                self.last_intra_conv_lig = TPCL(
                    args,
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=irrep_seq[-1],
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=False,
                ) 

                self.last_inter_conv_lig = TPCL(
                    args,
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=irrep_seq[-1],
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=False, 
                    is_last_layer=True,
                    is_layerNorm_edgeFC=args.readOutInterAttentionFeature or args.allInterAttentionFeature,
                ) 
            else:
                self.last_intra_conv_lig = self.last_intra_conv_rec
                self.last_inter_conv_lig = self.last_inter_conv_rec


        # center of mass translation and rotation components
        # (0.0, 30, 32)
        self.center_dist_exp = GaussianSmearing(
            0.0, self.center_max_dist, args.dist_embed_dim
        )

        if not args.layer_norm:
            self.center_edge_embed = nn.Sequential(
                nn.Linear(args.dist_embed_dim + args.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(ns, ns),
            )
        else:
            self.center_edge_embed = nn.Sequential(
                nn.Linear(args.dist_embed_dim + args.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.LayerNorm(ns, eps=args.layer_norm_eps),
                nn.Dropout(args.dropout),
                nn.Linear(ns, ns),
            )

        # global prediction
        self.final_conv = TPCL(
                args,
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f"{ns}x0e + 2x1o + 2x1e",
                n_edge_features=2 * ns,
                residual=False,
                is_last_layer=True,
            ) # (n_res, 12)

        # lddt, R, t
        self.lddt_final_layer = nn.Sequential(nn.Linear(ns, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1), nn.Sigmoid())
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))
        # flex
        self.rec_tr_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.lig_tr_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rec_res_rot_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.lig_res_rot_final_layer = nn.Sequential(nn.Linear(1 + args.sigma_embed_dim, ns),nn.Dropout(args.dropout), nn.ReLU(), nn.Linear(ns, 1))



        # output rec coords
        self.final_conv_node_wise_rec = TPCL(
            args,
            in_irreps=self.intra_convs[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f"1x1o + 1x1e",
            n_edge_features=3 * ns,
            hidden_features=3 * ns,
            residual=False,
            is_last_layer=True,
        ) # (B, res_num, 6)

        # output lig coords
        self.final_conv_node_wise_lig = TPCL(
            args,
            in_irreps=self.intra_convs[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f"1x1o + 1x1e",
            n_edge_features=3 * ns,
            hidden_features=3 * ns,
            residual=False,
            is_last_layer=True,
        ) # (B, res_num, 6)

        _init(self)

    def forward(self, batch, eps=1e-6):

        ### DEBUG
        DEBUG = False

        # get noise schedule
        t_tr = batch.complex_t["tr"]
        t_rot = batch.complex_t["rot"]
        t_flex_rec = batch.complex_t["flex"]
        t_flex_lig = batch.complex_t["flex"]


        # get sigma
        if self.args.adaptive_schedule:
            # print("batch.t_fm_deform", batch.t_fm_deform)
            # print("batch.adaptive_c", batch.adaptive_c)
            flex_rec_s, flex_lig_s, rot_s, tr_s = self.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr, adaptive_c=batch.adaptive_c)
        else:
            flex_rec_s, flex_lig_s, rot_s, tr_s = self.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)
        

        # build ligand graph
        ligand_graph = self.build_radius_graph(batch, "ligand") # return node_attr, edge_index, edge_attr, edge_sh
        lig_node_attr = self.atom_embed(ligand_graph[0]) # lig_node_attr: shape (n_res, self.ns) FOR further embedding + reduce dimension [inv]
        lig_src, lig_dst = lig_edge_index = ligand_graph[1] # lig_edge_index: shape (2, n_edge)
        
        # use rec_edge_embed for both rec/lig
        lig_edge_attr = self.rec_edge_embed(ligand_graph[2]) # lig_edge_attr: (n_edge, self.ns) [inv]
        lig_edge_sh = ligand_graph[3] # lig_edge_sh: (n_edge, 9) [equi]

        # build receptor graph
        receptor_graph = self.build_radius_graph(batch, "receptor")
        rec_node_attr = self.atom_embed(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr = self.rec_edge_embed(receptor_graph[2])
        rec_edge_sh = receptor_graph[3]

        # build cross graph
        if self.dynamic_max_cross:
            # cross_cutoff is controlled by translation sigma
            # self.cross_cutoff_weight is defaulted as 3  
            # self.cross_cutoff_bias is defaulted as 40
            cross_cutoff = (tr_s * self.cross_cutoff_weight + self.cross_cutoff_bias).unsqueeze(1) # (bs, 1)
        else:
            cross_cutoff = self.cross_max_dist
        
        cross_edge_index, cross_edge_attr, cross_edge_sh, inv_cross_edge_sh = self.cross_conv_graph(
            batch, cross_cutoff
        )

        
        cross_lig, cross_rec = cross_edge_index # cross_lig: (n_edge,); cross_rec: (n_edge,)
        cross_edge_attr = self.cross_edge_embed(cross_edge_attr) # (n_edge, self.ns)


        # if self.args.intra_equi_nm_node_f:

        #     lig_node_attr_ = torch.cat([
        #         lig_node_attr,
        #         batch['ligand'].cur_nm.permute((0,2,1)).reshape((-1, 3*self.args.n_NM))
        #     ], -1)

        #     rec_node_attr_ = torch.cat([
        #         rec_node_attr,
        #         batch['receptor'].cur_nm.permute((0,2,1)).reshape((-1, 3*self.args.n_NM))
        #     ], -1)
        # else:
        #     lig_node_attr_ = lig_node_attr
        #     rec_node_attr_ = rec_node_attr

        for idx in range(len(self.intra_convs)):
            # if idx != 0 or (not self.args.intra_equi_nm_node_f):
            #     lig_node_attr_ = lig_node_attr
            #     rec_node_attr_ = rec_node_attr

            if self.args.res_rot_flex and idx == 0:
                rec_n_vec = batch['receptor'].bb_atoms[:,0] - batch['receptor'].bb_atoms[:,1]
                rec_n_norm_vec = rec_n_vec / (rec_n_vec.norm(dim=-1,keepdim=True)+eps)
                rec_c_vec = batch['receptor'].bb_atoms[:,2] - batch['receptor'].bb_atoms[:,1]
                rec_c_norm_vec = rec_c_vec / (rec_c_vec.norm(dim=-1,keepdim=True)+eps)

                lig_n_vec = batch['ligand'].bb_atoms[:,0] - batch['ligand'].bb_atoms[:,1]
                lig_n_norm_vec = lig_n_vec / (lig_n_vec.norm(dim=-1,keepdim=True)+eps)
                lig_c_vec = batch['ligand'].bb_atoms[:,2] - batch['ligand'].bb_atoms[:,1]
                lig_c_norm_vec = lig_c_vec / (lig_c_vec.norm(dim=-1,keepdim=True)+eps)

                rec_node_attr_ = torch.cat([rec_node_attr,rec_n_norm_vec,rec_c_norm_vec],dim=-1)
                lig_node_attr_ = torch.cat([lig_node_attr,lig_n_norm_vec,lig_c_norm_vec],dim=-1)
            else:
                lig_node_attr_ = lig_node_attr
                rec_node_attr_ = rec_node_attr
                
            # message passing within ligand graph (intra)
            lig_edge_attr_ = torch.cat([
                    lig_edge_attr,
                    lig_node_attr[lig_src, :self.ns],
                    lig_node_attr[lig_dst, :self.ns]], -1) # (n_edge, self.ns * 3)
            
            # lig_intra_update (n_res_lig, out_irreps_size)
            if self.args.interModuleNoSharedWeight:
                lig_intra_update = self.lig_intra_convs[idx](
                    lig_node_attr_, lig_edge_index,
                    lig_edge_attr_, lig_edge_sh)
            else:
                lig_intra_update = self.intra_convs[idx](
                    lig_node_attr_, lig_edge_index,
                    lig_edge_attr_, lig_edge_sh)

            # message passing between two graphs (inter)
            # rec2lig_edge_attr_ = torch.cat([
            #         cross_edge_attr,
            #         lig_node_attr[cross_lig, :self.ns], 
            #         rec_node_attr[cross_rec, :self.ns]], -1) # (n_edge, self.ns * 3)
            
            if self.args.allInterAttentionFeature:
                # (query, key/value, edge_index(key to query))
                rec2lig_attention_edge_attr = self.interModuleInterAttentions[idx](lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns], torch.flip(cross_edge_index, dims=[0])) # ignorant of the edge vectors
                rec2lig_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        rec2lig_attention_edge_attr], -1) # (n_edge, self.ns * 3)
            else:
                rec2lig_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        lig_node_attr[cross_lig, :self.ns], 
                        rec_node_attr[cross_rec, :self.ns]], -1) # (n_edge, self.ns * 3)
            
            # (n_res_lig, out_irreps_size)
            if self.args.interModuleNoSharedWeight:
                lig_inter_update = self.lig_cross_convs[idx](
                    rec_node_attr_,
                    cross_edge_index,
                    rec2lig_edge_attr_,
                    cross_edge_sh,
                    out_nodes=lig_node_attr.shape[0])
            else:
                lig_inter_update = self.cross_convs[idx](
                    rec_node_attr_,
                    cross_edge_index,
                    rec2lig_edge_attr_,
                    cross_edge_sh,
                    out_nodes=lig_node_attr.shape[0]) 

            # [TODO: do we still leave out the last layer's update for rec?]
            # message passing within receptor graph (intra)
            if (not self.args.recLigMismatchingInterLayer) or (idx != len(self.intra_convs) - 1):
                rec_edge_attr_ = torch.cat([
                        rec_edge_attr,
                        rec_node_attr[rec_src, :self.ns],
                        rec_node_attr[rec_dst, :self.ns]], -1) # (n_edge, self.ns * 3)
                
                # (n_res_rec, out_irreps_size)
                rec_intra_update = self.intra_convs[idx](
                    rec_node_attr_, rec_edge_index,
                    rec_edge_attr_, rec_edge_sh)
                
                # lig2rec_edge_attr_ = torch.cat([
                #         cross_edge_attr,
                #         rec_node_attr[cross_rec, : self.ns], # previously flipped
                #         lig_node_attr[cross_lig, : self.ns]], -1) # (n_edge, self.ns * 3)

                if self.args.allInterAttentionFeature:
                    # (query, key/value, edge_index(key to query))
                    lig2rec_attention_edge_attr = self.interModuleInterAttentions[idx](rec_node_attr[cross_rec, :self.ns], lig_node_attr[cross_lig, :self.ns], cross_edge_index)
                    lig2rec_edge_attr_ = torch.cat([
                            cross_edge_attr,
                            lig2rec_attention_edge_attr], -1)
                else:
                    lig2rec_edge_attr_ = torch.cat([
                            cross_edge_attr,
                            rec_node_attr[cross_rec, : self.ns], # previously flipped
                            lig_node_attr[cross_lig, : self.ns]], -1)
                    
                rec_inter_update = self.cross_convs[idx](
                    lig_node_attr_,
                    torch.flip(cross_edge_index, dims=[0]), # rec -> lig
                    lig2rec_edge_attr_,
                    inv_cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0])

                # padding original features
                rec_node_attr = F.pad(
                    rec_node_attr,
                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                # update features with residual updates
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

            # padding original features
            lig_node_attr = F.pad(
                lig_node_attr,
                (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            
            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if idx == len(self.intra_convs) - 2:
                secondLastLayer_rec_node_attr = rec_node_attr
                secondLastLayer_lig_node_attr = lig_node_attr
        #
                
        #
        # addtional one layer, if self.args.indReadoutFlex_sharedWeight or self.args.indReadoutFlex_noSharedWeight
        if self.args.indReadoutFlex_sharedWeight or self.args.indReadoutFlex_noSharedWeight:
            # last_intra_conv_rec

            lig_node_attr_ = secondLastLayer_lig_node_attr
            rec_node_attr_ = secondLastLayer_rec_node_attr
            
            # ligand
            # message passing within ligand graph (intra)
            lig_edge_attr_ = torch.cat([
                    lig_edge_attr,
                    lig_node_attr_[lig_src, :self.ns],
                    lig_node_attr_[lig_dst, :self.ns]], -1) # (n_edge, self.ns * 3)
            # lig_intra_update (n_res_lig, out_irreps_size)
            lig_intra_update = self.last_intra_conv_lig(
                lig_node_attr_, lig_edge_index,
                lig_edge_attr_, lig_edge_sh)

            # message passing between two graphs (inter)
            if self.args.readOutInterAttentionFeature or self.args.allInterAttentionFeature:
                # (query, key/value, edge_index(key to query))
                rec2lig_attention_edge_attr = self.readOutInterAttention(lig_node_attr_[cross_lig, :self.ns], rec_node_attr_[cross_rec, :self.ns], torch.flip(cross_edge_index, dims=[0])) # ignorant of the edge vectors
                rec2lig_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        rec2lig_attention_edge_attr], -1) # (n_edge, self.ns * 3)
            else:
                rec2lig_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        lig_node_attr_[cross_lig, :self.ns], 
                        rec_node_attr_[cross_rec, :self.ns]], -1) # (n_edge, self.ns * 3)
            # (n_res_lig, out_irreps_size)
            lig_inter_update = self.last_inter_conv_lig(
                rec_node_attr_,
                cross_edge_index,
                rec2lig_edge_attr_,
                cross_edge_sh,
                out_nodes=lig_node_attr_.shape[0]) 

            # receptor
            rec_edge_attr_ = torch.cat([
                    rec_edge_attr,
                    rec_node_attr_[rec_src, :self.ns],
                    rec_node_attr_[rec_dst, :self.ns]], -1) # (n_edge, self.ns * 3)
            # (n_res_rec, out_irreps_size)
            rec_intra_update = self.last_intra_conv_rec(
                rec_node_attr_, rec_edge_index,
                rec_edge_attr_, rec_edge_sh)
            
            if self.args.readOutInterAttentionFeature or self.args.allInterAttentionFeature:
                # (query, key/value, edge_index(key to query))
                lig2rec_attention_edge_attr = self.readOutInterAttention(rec_node_attr_[cross_rec, :self.ns], lig_node_attr_[cross_lig, :self.ns], cross_edge_index)
                lig2rec_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        lig2rec_attention_edge_attr], -1)
            else:
                lig2rec_edge_attr_ = torch.cat([
                        cross_edge_attr,
                        rec_node_attr_[cross_rec, : self.ns], # previously flipped
                        lig_node_attr_[cross_lig, : self.ns]], -1)
                
            rec_inter_update = self.last_inter_conv_rec(
                lig_node_attr_,
                torch.flip(cross_edge_index, dims=[0]), # rec -> lig
                lig2rec_edge_attr_,
                inv_cross_edge_sh,
                out_nodes=rec_node_attr_.shape[0])

            # padding original features
            rec_node_attr_ = F.pad(
                rec_node_attr_,
                (0, rec_intra_update.shape[-1] - rec_node_attr_.shape[-1]))
            # update features with residual updates
            rec_node_attr_final = rec_node_attr_ + rec_intra_update + rec_inter_update

            # padding original features
            lig_node_attr_ = F.pad(
                lig_node_attr_,
                (0, lig_intra_update.shape[-1] - lig_node_attr_.shape[-1]))
            
            # update features with residual updates
            lig_node_attr_final = lig_node_attr_ + lig_intra_update + lig_inter_update
        

            

        #
        # construct centered ligand graph
        (
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
        ) = self.center_conv_graph(batch, "ligand")
        center_edge_attr = self.center_edge_embed(center_edge_attr) # (n_edge, ns)

        # [Wrong before: center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], : self.ns]], -1)
        center_edge_attr = torch.cat(
            [center_edge_attr, lig_node_attr[center_edge_index[1], : self.ns]], -1
        ) # (n_edge, ns * 2) # [???? which node]

        #
        # lddt prediction (global prediction)
        global_pred = self.final_conv(
            lig_node_attr,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=batch.num_graphs,
        ) # (num_graphs, self.ns+12)

        lddt_pred = self.lddt_final_layer(global_pred[:,:self.ns])

        tr_pred = (global_pred[:, self.ns:self.ns+3] + global_pred[:, self.ns+3:self.ns+6])
        rot_pred = global_pred[:, self.ns+6:self.ns+9] + global_pred[:, self.ns+9:]
        # print(tr_pred)
        # print(rot_pred)
        batch_sigma_emb = self.t_embedding(batch.complex_t["tr"])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / (tr_norm+eps) * self.tr_final_layer(torch.cat([tr_norm, batch_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / (rot_norm+eps) * self.rot_final_layer(torch.cat([rot_norm, batch_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred * tr_s.unsqueeze(1)
            rot_pred = rot_pred * rot_s.unsqueeze(1)#so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)




        if self.args.recLigMismatchingInterLayer:

            if self.args.secondLastLayerLigFlex or self.args.secondLastLayerLigFlex_halfl1:
                lig_node_attr_final = secondLastLayer_lig_node_attr
                rec_node_attr_final = secondLastLayer_rec_node_attr
            elif self.args.indReadoutFlex_sharedWeight or self.args.indReadoutFlex_noSharedWeight: # this is the case
                lig_node_attr_final = lig_node_attr_final
                rec_node_attr_final = rec_node_attr_final
            else:
                lig_node_attr_final = lig_node_attr
                rec_node_attr_final = rec_node_attr        
            
            if self.args.res_rot_flex:
                delta_rec_coord = (rec_node_attr_final[:, self.ns:self.ns+self.nv*3]).view(rec_node_attr_final.shape[0],-1,3).mean(1)
                delta_lig_coord = (lig_node_attr_final[:, self.ns:self.ns+self.nv*3]).view(lig_node_attr_final.shape[0],-1,3).mean(1)
                delta_rec_rot = (rec_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*6]).view(rec_node_attr_final.shape[0],-1,3).mean(1)
                delta_lig_rot = (lig_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*6]).view(lig_node_attr_final.shape[0],-1,3).mean(1)
            elif not self.args.secondLastLayerLigFlex_halfl1:
                # read out the translation vectors
                # delta_rec_coord = (rec_node_attr_final[:, self.ns:self.ns+self.nv*3]).view(rec_node_attr_final.shape[0],-1,3).mean(1)
                # delta_lig_coord = (lig_node_attr_final[:, self.ns:self.ns+self.nv*3]).view(lig_node_attr_final.shape[0],-1,3).mean(1)
                delta_rec_coord = (rec_node_attr_final[:, self.ns:self.ns+self.nv*3]+rec_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*6]).view(rec_node_attr_final.shape[0],-1,3).mean(1)
                delta_lig_coord = (lig_node_attr_final[:, self.ns:self.ns+self.nv*3]+lig_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*6]).view(lig_node_attr_final.shape[0],-1,3).mean(1)
            else:
                delta_rec_coord = (rec_node_attr_final[:, self.ns:self.ns+self.nv*3//2]+rec_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*3+self.nv*3//2]).view(rec_node_attr_final.shape[0],-1,3).mean(1)
                delta_lig_coord = (lig_node_attr_final[:, self.ns:self.ns+self.nv*3//2]+lig_node_attr_final[:, self.ns+self.nv*3:self.ns+self.nv*3+self.nv*3//2]).view(lig_node_attr_final.shape[0],-1,3).mean(1)
            


            # magnitude prediction
            if self.args.adaptive_schedule: # [adaptive_schedule]
                rec_tr_t_embed = self.t_embedding(batch["receptor"].t_fm_deform)
            else:
                rec_tr_t_embed = self.t_embedding(batch["receptor"].node_t["tr"])
            delta_rec_coord_norm = torch.linalg.vector_norm(delta_rec_coord, dim=1)[:, None] # (num_graphs, 1)
            delta_rec_coord_scale = self.rec_tr_final_layer(
                torch.cat([delta_rec_coord_norm, rec_tr_t_embed], dim=1)) # (num_graphs, 1)
            delta_rec_coord = (delta_rec_coord / (delta_rec_coord_norm+eps)) * delta_rec_coord_scale # (num_graphs, 3)

            if self.args.adaptive_schedule: # [adaptive_schedule]
                lig_tr_t_embed = self.t_embedding(batch["ligand"].t_fm_deform)
            else:
                lig_tr_t_embed = self.t_embedding(batch["ligand"].node_t["tr"])
            delta_lig_coord_norm = torch.linalg.vector_norm(delta_lig_coord, dim=1)[:, None] # (num_graphs, 1)
            delta_lig_coord_scale = self.lig_tr_final_layer(
                torch.cat([delta_lig_coord_norm, lig_tr_t_embed], dim=1)) # (num_graphs, 1)
            delta_lig_coord = (delta_lig_coord / (delta_lig_coord_norm+eps)) * delta_lig_coord_scale # (num_graphs, 3)

            if self.args.res_rot_flex:
                if self.args.adaptive_schedule: # [adaptive_schedule]
                    rec_rot_t_embed = self.t_embedding(batch["receptor"].t_fm_deform)
                else:
                    rec_rot_t_embed = self.t_embedding(batch["receptor"].node_t["rot"])
                delta_rec_rot_norm = torch.linalg.vector_norm(delta_rec_rot, dim=1)[:, None] # (num_graphs, 1)
                delta_rec_rot_scale = self.rec_res_rot_final_layer(
                    torch.cat([delta_rec_rot_norm, rec_rot_t_embed], dim=1))
                delta_rec_rot = (delta_rec_rot / (delta_rec_rot_norm+eps)) * delta_rec_rot_scale

                if self.args.adaptive_schedule: # [adaptive_schedule]
                    lig_rot_t_embed = self.t_embedding(batch["ligand"].t_fm_deform)
                else:
                    lig_rot_t_embed = self.t_embedding(batch["ligand"].node_t["rot"])
                delta_lig_rot_norm = torch.linalg.vector_norm(delta_lig_rot, dim=1)[:, None] # (num_graphs, 1)
                delta_lig_rot_scale = self.lig_res_rot_final_layer(
                    torch.cat([delta_lig_rot_norm, lig_rot_t_embed], dim=1))
                delta_lig_rot = (delta_lig_rot / (delta_lig_rot_norm+eps)) * delta_lig_rot_scale

                return delta_rec_coord, delta_lig_coord, lddt_pred, rot_pred, tr_pred, delta_rec_rot, delta_lig_rot

            return delta_rec_coord, delta_lig_coord, lddt_pred, rot_pred, tr_pred
        

        #
        # coord prediction (residue prediction)
        # [TODO: there is no explicit NM node feature / NM edge feature here]
        # final update to get delta coords for r/l
        # message passing within ligand graph (intra)
        lig_edge_attr_ = torch.cat([
                lig_edge_attr,
                lig_node_attr[lig_src, :self.ns],
                lig_node_attr[lig_dst, :self.ns]], -1) # (n_edge, self.ns * 3)
        # lig_intra_update (n_res_lig, 6)
        delta_lig_coord = self.final_conv_node_wise_lig(
            lig_node_attr, lig_edge_index,
            lig_edge_attr_, lig_edge_sh)

        # message passing within receptor graph (intra)
        rec_edge_attr_ = torch.cat([
                rec_edge_attr,
                rec_node_attr[rec_src, :self.ns],
                rec_node_attr[rec_dst, :self.ns]], -1)
        # rec_intra_update (n_res_rec, 6)
        delta_rec_coord = self.final_conv_node_wise_rec(
            rec_node_attr, rec_edge_index,
            rec_edge_attr_, rec_edge_sh)

        delta_rec_coord = delta_rec_coord[...,:3] + delta_rec_coord[...,3:6]
        delta_lig_coord = delta_lig_coord[...,:3] + delta_lig_coord[...,3:6]

        # update the batch object
        batch['receptor'].delta_pos = delta_rec_coord
        batch['ligand'].delta_pos = delta_lig_coord

        return delta_rec_coord, delta_lig_coord, lddt_pred, rot_pred, tr_pred

        
       
        

    def build_radius_graph(self, batch, key):
        """
            proteins radius graph
            Adds noise information to existing embeddings.
            @return node_attr, edge_index, edge_attr, edge_sh
        """
        DEBUG = False
        # batch[key].node_sigma_emb.shape == (n_res, self.t_embed_dim=32)
        batch[key].node_sigma_emb = self.t_embedding( 
            batch[key].node_t["tr"]
        )  # tr rot and nm noise is all the same

        if DEBUG:
            print('batch[key].node_sigma_emb.shape', batch[key].node_sigma_emb.shape)
        
        # if esm, batch[key].x.shape == (n_res, self.esm_embed_dim=1281)
        # if no esm, batch[key].x.shape == (n_res, 1)
        # the first entry in dim 0 of batch[key].x is the aa type

        # if no ESM models, graph.x should still be flat
        if len(batch[key].x.shape) == 1:
            batch[key].x = batch[key].x[:,None]
        
        # node_attr.shape == (n_res, self.esm_embed_dim=1281 + self.t_embed_dim=32)
        node_attr = torch.cat(
            [batch[key].x, batch[key].node_sigma_emb], 1
        )
        if DEBUG:
            print('node_attr.shape', node_attr.shape)
        
        #
        # builds the graph edges and initial node and edge features

        #
        # compute edges
        radius_edges = radius_graph(batch[key].pos, self.intra_max_radius, batch[key].batch) #### [?]
        if DEBUG:
            print('radius_edges.shape', radius_edges.shape)
        
        # batch[key, key].edge_index should be (2, n_res)
        edge_index = radius_edges.long()
        if DEBUG:
            print('edge_index.shape', edge_index.shape)

        if DEBUG_cutoff:
            print('self.intra_max_radius', self.intra_max_radius)
            print('node_attr.shape', node_attr.shape)
            print('radius_edges.shape', radius_edges.shape)
            print('edge_index.shape', edge_index.shape)

        
        # # this assumes the edges were already created in preprocessing since protein's structure is fixed
        # edge_index = batch[key, key].edge_index
        src, dst = edge_index
        edge_vec = batch[key].pos[dst.long()] - batch[key].pos[src.long()] # [n_edge, 3]
        if DEBUG:
            print('edge_vec.shape', edge_vec.shape)

        edge_length_emb = self.rec_dist_exp(edge_vec.norm(dim=-1)) # [n_edge, n_gauss=32]
        if DEBUG:
            print('edge_length_emb.shape', edge_length_emb.shape)

        edge_sigma_emb = batch[key].node_sigma_emb[edge_index[0].long()] # [n_edge, self.t_embed_dim] for the embedding of the edge's source node
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1) # [n_edge, self.t_embed_dim=32 + n_gauss=32]
        if DEBUG:
            print('edge_attr.shape', edge_attr.shape)

        # edge_sh.shape: (n_edge, 9)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec,
            normalize=True, normalization="component")
        if DEBUG:
            print('edge_sh.shape', edge_sh.shape)

        if self.args.use_nm_mode_correlation_feat:
            normal_mode_matrix = batch[key].cur_nm
            eigenvalues = batch[key].eig_vals

            # print('batch[key].cur_nm', batch[key].cur_nm.shape)
            # print('batch[key].eig_vals', batch[key].eig_vals.shape)

            nm_correlation_embd = self.nm_corr_grb_embed(calculate_mode_correlation(edge_index, normal_mode_matrix, eigenvalues, batch[key].batch))
            
            edge_attr = torch.cat([edge_attr, nm_correlation_embd], 1) # [n_edge, self.t_embed_dim=32 + n_gauss=32 + self.nm_corr_embed_dim=16]
        
        if self.args.use_nm_msf_feat:
            # print('batch[key].msf', batch[key].msf.shape, 'node_attr', node_attr.shape)
            # node_attr = torch.cat([node_attr, self.t_embedding(batch[key].msf)], 1) # [n_res, self.esm_embed_dim=1281 + self.t_embed_dim=32 + self.msf_embed_dim=32]
            node_attr = torch.cat([node_attr, self.nm_msf_grb_embed(batch[key].msf)], 1) # [n_res, self.esm_embed_dim=1281 + self.t_embed_dim=32 + self.msf_embed_dim=16]

        # node_attr.shape: (n_res, self.esm_embed_dim + self.t_embed_dim (+ self.msf_embed_dim=16))
        # edge_index.shape: (2, n_res)
        # edge_attr.shape: (n_edge, self.t_embed_dim=32 + n_gauss=32 (+ self.nm_corr_embed_dim=16))
        # edge_sh.shape: (n_edge, 9)
        return node_attr, edge_index, edge_attr, edge_sh 

    def cross_conv_graph(self, batch, cross_dist_cutoff):
        """
            Builds the cross edges between ligand and receptor
            @return edge_index, edge_attr, edge_sh
        """

        DEBUG = False

        if DEBUG:
            # print('batch["receptor"].pos.shape', batch["receptor"].pos.shape)
            # print('batch["ligand"].pos.shape', batch["ligand"].pos.shape)
            # print('batch["ligand"].batch.shape', batch["ligand"].batch.shape)
            # print('batch["receptor"].batch.shape', batch["receptor"].batch.shape)
            # print('cross_dist_cutoff.shape', cross_dist_cutoff.shape)
            # print('batch["ligand"].batch', batch["ligand"].batch)
            pass
            
        if torch.is_tensor(cross_dist_cutoff):
            # different cutoff for every graph
            # (depends on the diffusion time)
            edge_index = radius(
                batch["receptor"].pos / cross_dist_cutoff[batch["receptor"].batch],
                batch["ligand"].pos / cross_dist_cutoff[batch["ligand"].batch],
                1,
                batch["receptor"].batch,
                batch["ligand"].batch,
                max_num_neighbors=self.max_num_cross_neighbors,
            )
        else:
            edge_index = radius(
                batch["receptor"].pos,
                batch["ligand"].pos,
                cross_dist_cutoff,
                batch["receptor"].batch,
                batch["ligand"].batch,
                max_num_neighbors=self.max_num_cross_neighbors,
            )
        
        if DEBUG_cutoff:
            print('cross_dist_cutoff', cross_dist_cutoff)
            print('edge_index.shape', edge_index.shape)

        src, dst = edge_index # [2, n_edge]
        # if DEBUG:
        #     print('edge_index.shape', edge_index.shape)

        # lig -> rec
        edge_vec = batch["receptor"].pos[dst.long()] - batch["ligand"].pos[src.long()] # (n_edge, 3)

        edge_length_emb = self.cross_dist_exp(edge_vec.norm(dim=-1)) # (n_edge, n_gauss=32)
        edge_sigma_emb = batch["ligand"].node_sigma_emb[src.long()] # (n_edge, self.t_embed_dim=32) # the embedding of ligand sigma (the same as receptor sigma)
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1) # (n_edge, self.t_embed_dim=32 + n_gauss=32)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec,
            normalize=True, normalization="component") # (n_edge, 9)
        
        inv_edge_sh = o3.spherical_harmonics(self.sh_irreps, -edge_vec,
            normalize=True, normalization="component") # (n_edge, 9)
        
        if DEBUG:
            print("edge_sh.shape", edge_sh.shape)
        
        # edge_index shape (2, n_edge)
        # edge_attr shape (n_edge, self.t_embed_dim=32 + n_gauss=32)
        # edge_sh shape (n_edge, 9)
        # inv_edge_sh shape (n_edge, 9)
        return edge_index, edge_attr, edge_sh, inv_edge_sh

    # [TODO: how the init edge_index is calculated?]
    def center_conv_graph(self, batch, key):
        """
            Builds the filter and edges for the convolution
            @return edge_index, edge_attr, edge_sh
        """

        DEBUG = False

        #
        # The resulting edge_index tensor will have two rows: 
        # the first row is from batch["ligand"].batch 
        # and the second row is a sequence of integers from 0 to len(batch["ligand"].batch) - 1.
        # (This structure is typical for representing edges in a graph, where each column of edge_index represents an edge, 
        # and the two elements in the column represent the nodes that the edge connects.)
        edge_index = torch.cat(
            [
                batch[key].batch.unsqueeze(0),
                torch.arange(len(batch[key].batch))
                .to(batch[key].x.device)
                .unsqueeze(0),
            ],
            dim=0,
        )

        # the first row of indices is the graph identifiers for each node
        # the second row of indices is the node index for each node

        if DEBUG:
            print('batch[key].batch.unsqueeze(0).shape', batch[key].batch.unsqueeze(0).shape)
            print('len(batch[key].batch)', len(batch[key].batch))
            
            
        
        #
        # compute the center of positions (presumably center of mass or centroid) for each graph in a batch of graphs.
        center_pos, count = torch.zeros((batch.num_graphs, 3)).to(
            batch[key].x.device
        ), torch.zeros((batch.num_graphs, 3)).to(batch[key].x.device)
        center_pos.index_add_(
            0, index=batch[key].batch, source=batch[key].pos
        )
        center_pos = center_pos / torch.bincount(batch[key].batch).unsqueeze(1)

        # a vector pointing from a graph's center position to a node
        edge_vec = batch[key].pos[edge_index[1]] - center_pos[edge_index[0]]
        # dist embedding
        # (n_edge, n_gauss=32)
        edge_attr = self.center_dist_exp(edge_vec.norm(dim=-1))
        # the node sigma embedding
        # (n_edge, t_embed_dim=32)
        edge_sigma_emb = batch[key].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True,
            normalization="component"
        )

        # edge_index shape (2, n_node)
        # edge_attr shape (n_node, t_embed_dim=32 + n_gauss=32)
        # edge_sh shape (n_node, 9)
        return edge_index, edge_attr, edge_sh # [[center, node]], edge_attr, edge_sh

    def bond_conv_graph(self, batch):
        """
            Builds the graph for the convolution between
            the center of the rotatable bonds and the neighbouring
            nodes
        """
        bonds = (
            batch["ligand", "ligand"].edge_index[:, batch["ligand"].edge_mask].long()
        )
        bond_pos = (batch["ligand"].pos[bonds[0]] + batch["ligand"].pos[bonds[1]]) / 2
        bond_batch = batch["ligand"].batch[bonds[0]]
        edge_index = radius(
            batch["ligand"].pos,
            bond_pos,
            self.max_radius_bond, # defaulted as 5A bt pos and bond_pos
            batch_x=batch["ligand"].batch,
            batch_y=bond_batch,
        )

        edge_vec = batch["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_dist_exp(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embed(edge_attr)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True,
            normalization="component"
        )

        return bonds, edge_index, edge_attr, edge_sh

#
class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)) # [n_edge, n_gauss]


def get_timestep_embedding(args):
    if args.embedding_type == "sinusoidal":
        emb_func = SinusoidalEmbedding(args)
    elif args.embedding_type == "fourier":
        emb_func = GaussianFourierProjection(args)
    else:
        raise NotImplemented
    return emb_func

#
class SinusoidalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.sigma_embed_dim
        self.scale = args.embedding_scale
        self.max_positions = 1e4

    def forward(self, x):
        """
        From https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        """
        assert len(x.shape) == 1
        x = self.scale * x

        half_dim = self.embed_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim,
                dtype=torch.float32,
                device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb

# 
class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.sigma_embed_dim
        self.scale = args.embedding_scale
        self.W = nn.Parameter(
            torch.randn(self.embed_dim // 2) * self.scale,
            requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)
        return emb
    



def calculate_mode_correlation(edge_index, normal_mode_matrix, eigenvalues, batch):
    """
    Calculate mode correlation values for each edge in a batch-aware graph.

    Args:
        edge_index (torch.Tensor): Shape (2, n_edges), source and destination indices for each edge.
        normal_mode_matrix (torch.Tensor): Shape (total_node_num, 3, n_NM), normal modes for each node.
        eigenvalues (torch.Tensor): Shape (batch_size, n_NM), eigenvalues for each batch.
        batch (torch.Tensor): Shape (total_node_num,), batch assignment for each node.

    Returns:
        torch.Tensor: Mode correlation values for each edge, shape (n_edges,).
    """
    src, dst = edge_index  # Source and destination node indices
    n_NM = normal_mode_matrix.shape[-1]  # Number of modes

    # Map batch-specific eigenvalues to each edge
    src_batch = batch[src]  # Shape: (n_edges,)
    dst_batch = batch[dst]  # Shape: (n_edges,)

    # Ensure source and destination nodes belong to the same batch
    assert torch.equal(src_batch, dst_batch), "Source and destination nodes must belong to the same batch"
    edge_batch = src_batch

    # Extract batch-specific eigenvalues
    inv_eigenvalues = 1.0 / eigenvalues[edge_batch]  # Shape: (n_edges, n_NM)

    # Extract displacements for source and destination nodes
    u_src = normal_mode_matrix[src]  # Shape: (n_edges, 3, n_NM)
    u_dst = normal_mode_matrix[dst]  # Shape: (n_edges, 3, n_NM)

    # Compute the dot product for each edge and mode: u_i^m · u_j^m
    dot_product = torch.einsum("eim,eim->em", u_src, u_dst)  # Shape: (n_edges, n_NM)

    # Compute weighted dot product (numerator)
    numerator = torch.sum(dot_product * inv_eigenvalues, dim=-1)  # Shape: (n_edges,)

    # Compute the squared norms for source and destination nodes
    norm_src = torch.einsum("eim,eim->em", u_src, u_src)  # Shape: (n_edges, n_NM)
    norm_dst = torch.einsum("eim,eim->em", u_dst, u_dst)  # Shape: (n_edges, n_NM)

    # Compute weighted norms
    weighted_norm_src = torch.sum(norm_src * inv_eigenvalues, dim=-1)  # Shape: (n_edges,)
    weighted_norm_dst = torch.sum(norm_dst * inv_eigenvalues, dim=-1)  # Shape: (n_edges,)

    # Compute the denominator
    denominator = torch.sqrt(weighted_norm_src * weighted_norm_dst)  # Shape: (n_edges,)

    # Compute mode correlation
    correlation = numerator / (denominator + 1e-8)  # Add epsilon to avoid division by zero

    return correlation







