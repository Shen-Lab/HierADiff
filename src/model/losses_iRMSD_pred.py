import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from geom_utils import NoiseSchedule_flexFM_Rt, score_norm, rigid_transform_Kabsch_3D_torch_batch, kabsch_torch
from geom_utils.geometry import axis_angle_to_matrix


import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class compute_total_iRMSD_pred_loss(nn.Module):
    def __init__(self, args):
        super(compute_total_iRMSD_pred_loss, self).__init__()
        self.args = args
    
    def forward(self, model_out, batch, loss_weights={'irmsd': 1.0, 'contrastive':1.0, 'interface': 1.0}, contact_threshold=8.0):
        """
        Standalone loss function for iRMSD prediction and interface coordinate regularization.
        
        Args:
            model_out (dict): Output from the IEGMN model, expected to contain:
                - 'receptor': dict with keys:
                    'pos': predicted receptor positions, shape (N_rec, 3)
                    'batch': receptor batch indices, shape (N_rec,)
                    'pos_b_ori': ground-truth receptor backbone positions, shape (N_rec, 3)
                - 'ligand': dict with keys:
                    'pos': predicted ligand positions, shape (N_lig, 3)
                    'batch': ligand batch indices, shape (N_lig,)
                    'pos_b_ori': ground-truth ligand backbone positions, shape (N_lig, 3)
                - 'irmsd_pred': predicted iRMSD per graph, shape (num_graphs, 1)
            batch (dict): Batch dictionary that must contain:
                - batch['irmsd_label']: ground-truth iRMSD per graph, shape (num_graphs, 1)
            loss_weights (dict): Weights for each loss component (keys: 'irmsd' and 'coord').
            contact_threshold (float): Distance threshold (in Å) to define an interface (default: 8.0 Å).
            
        Returns:
            dict: Dictionary with individual losses and total loss.
        """
        # ------------------------------------------
        # 1. iRMSD Prediction Loss (MSE)
        # ------------------------------------------
        gt_irmsd = batch['irmsd']           # (num_graphs, 1)
        irmsd_pred = model_out['irmsd_pred']        # (num_graphs, 1)
        # clamp iRMSD predictions to [0.35, 8.0] Å
        irmsd_pred = torch.clamp(irmsd_pred, min=0.35, max=8.0)
        irmsd_loss = F.mse_loss(irmsd_pred, gt_irmsd)

        
        print("irmsd_pred: ", irmsd_pred, "gt_irmsd: ", gt_irmsd)

        if self.args.interface_reg:
            pred_interface_rec = model_out['receptor']['interface_pred']
            pred_interface_lig = model_out['ligand']['interface_pred']
            interface_mask_rec = batch['receptor'].interface_mask.float()
            interface_mask_lig = batch['ligand'].interface_mask.float()


            # For ligand:
            num_pos_lig = interface_mask_lig.sum()
            num_neg_lig = interface_mask_lig.numel() - num_pos_lig
            # Avoid division by zero:
            pos_weight_lig = num_neg_lig / (num_pos_lig + 1e-8)
            # Create a weight tensor: positive samples get pos_weight_lig, negatives get 1.
            weight_lig = interface_mask_lig * pos_weight_lig + (1 - interface_mask_lig)
            loss_interface_lig = F.binary_cross_entropy(pred_interface_lig, interface_mask_lig, weight=weight_lig)

            # For receptor:
            num_pos_rec = interface_mask_rec.sum()
            num_neg_rec = interface_mask_rec.numel() - num_pos_rec
            pos_weight_rec = num_neg_rec / (num_pos_rec + 1e-8)
            weight_rec = interface_mask_rec * pos_weight_rec + (1 - interface_mask_rec)
            loss_interface_rec = F.binary_cross_entropy(pred_interface_rec, interface_mask_rec, weight=weight_rec)
            
            
            interface_loss = loss_interface_lig + loss_interface_rec

        if self.args.contrastive:
            # contrastive loss
            contr_loss = contrastive_loss(irmsd_pred, gt_irmsd, margin=1.0, threshold_sim=0.5)
            print("contr_loss: ", contr_loss)


        if self.args.interface_reg and self.args.contrastive:
            total_loss = loss_weights['irmsd'] * irmsd_loss + loss_weights['interface'] * interface_loss + loss_weights['contrastive'] * contr_loss
            return {
                'irmsd_loss': irmsd_loss,
                'interface_loss': interface_loss,
                'contr_loss': contr_loss,
                'loss': total_loss
            } 

        elif not self.args.interface_reg and self.args.contrastive:

            total_loss = loss_weights['irmsd'] * irmsd_loss + loss_weights['contrastive'] * contr_loss

            return {
                'irmsd_loss': irmsd_loss,
                'contr_loss': contr_loss,
                'loss': total_loss
            }
        
        elif self.args.interface_reg and not self.args.contrastive:
            total_loss = loss_weights['irmsd'] * irmsd_loss + loss_weights['interface'] * interface_loss

            return {
                'irmsd_loss': irmsd_loss,
                'interface_loss': interface_loss,
                'loss': total_loss
            }
        
        else:
            total_loss = loss_weights['irmsd'] * irmsd_loss

            return {
                'irmsd_loss': irmsd_loss,
                'loss': irmsd_loss
            }


def contrastive_loss(irmsd_pred, irmsd_gt, margin=1.0, threshold_sim=0.5):
    """
    Computes a contrastive loss between pairs of predicted iRMSD values.
    
    For each pair (i, j) in the batch:
      - If |gt_i - gt_j| < threshold_sim (i.e. they are similar), we want the predicted difference 
        to be small (L = 0.5 * (pred_i - pred_j)^2).
      - Otherwise, we want the predicted difference to be at least the margin (L = 0.5 * max(0, margin - |pred_i - pred_j|)^2).
    
    Args:
        irmsd_pred (Tensor): Predicted iRMSD values of shape (B, 1) or (B,).
        irmsd_gt (Tensor): Ground truth iRMSD values of shape (B, 1) or (B,).
        margin (float): The minimum margin for dissimilar pairs.
        threshold_sim (float): The threshold under which two graphs are considered similar.
        
    Returns:
        Tensor: A scalar contrastive loss.
    """
    if irmsd_pred.dim() == 1:
        irmsd_pred = irmsd_pred.unsqueeze(1)
    if irmsd_gt.dim() == 1:
        irmsd_gt = irmsd_gt.unsqueeze(1)
        
    # Compute pairwise absolute differences.
    diff_pred = torch.abs(irmsd_pred - irmsd_pred.t())  # Shape: (B, B)
    diff_gt = torch.abs(irmsd_gt - irmsd_gt.t())          # Shape: (B, B)
    
    # Create masks for similar and dissimilar pairs.
    similar_mask = (diff_gt < threshold_sim).float()  # 1 if similar, 0 if not.
    dissimilar_mask = 1.0 - similar_mask
    
    # Contrastive loss terms.
    loss_sim = 0.5 * (diff_pred ** 2) * similar_mask
    loss_dissim = 0.5 * (F.relu(margin - diff_pred) ** 2) * dissimilar_mask
    
    # We average over the upper triangular part (excluding the diagonal).
    B = irmsd_pred.size(0)
    mask = torch.triu(torch.ones(B, B, device=irmsd_pred.device), diagonal=1)
    total_pairs = mask.sum() + 1e-6
    
    contrastive = ((loss_sim + loss_dissim) * mask).sum() / total_pairs
    return contrastive



    





