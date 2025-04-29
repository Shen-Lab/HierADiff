import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from geom_utils import NoiseSchedule_flexFM_Rt, score_norm, rigid_transform_Kabsch_3D_torch_batch, kabsch_torch
from geom_utils.geometry import axis_angle_to_matrix



def get_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(f"{msg} shape: {tensor.shape}"
                  f" max: {tensor.max()} min: {tensor.min()}"
                  f" mean: {tensor.mean()}")
    return printer


def register_hook(tensor, msg):
    """Utility function to call retain_grad and Pytorch's register_hook
    in a single line
    """
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))





# [Cartesian Space Flexing]
class flexFM_RtConfidenceLoss(nn.Module):
    def __init__(self, args):
        super(flexFM_RtConfidenceLoss, self).__init__()
        self.args = args
        
        self.lddt_weight = args.lddt_weight # 1
        self.irmsd_confidence_weight = args.irmsd_confidence_weight # 1
        
        self.eps = 1e-5

    def forward(self, data, outputs, apply_mean=True):
        """
            @param (dict) outputs
            @param (torch_geometric.data.HeteroData) data
        """

        


        DEBUG = False

        batch_data = data

        mean_dims = (0, 1) if apply_mean else 1

        # extract outputs
        
        
        lddt_pred = outputs["lddt_pred"]

        device = lddt_pred.device
        
        # lddt
        lddt = torch.cat([d.lddt for d in data], dim=0) if isinstance(data, list) else data.lddt

        # if self.args.lddt_loss_type == "l1":
        #     # Compute element-wise absolute error and then average over mean_dims
        #     lddt_loss = F.l1_loss(lddt_pred, lddt, reduction="none").mean(dim=mean_dims)
        #     # Compute absolute value of lddt using l1_loss with zero target, then average and detach
        #     lddt_base_loss = F.l1_loss(lddt, torch.zeros_like(lddt), reduction="none").mean(dim=mean_dims).detach()
        # elif self.args.lddt_loss_type == "l2":
        lddt_loss = ((lddt_pred - lddt) ** 2).mean(dim=mean_dims)
        lddt_base_loss = (lddt ** 2).mean(dim=mean_dims).detach()

        # print(f"lddt_loss: {lddt_loss}, lddt_base_loss: {lddt_base_loss}")

        if self.args.confidence_w_iRMSD_classification:
            irmsd_pred = outputs["iRMSD_classification"]


            irmsd_confidence = (
                torch.cat([d.irmsd_confidence for d in data], dim=0)
                if isinstance(data, list) else data.irmsd_confidence
            ).float()

            # Count positive and negative examples.
            num_pos = (irmsd_confidence == 1).sum().float()
            num_neg = (irmsd_confidence == 0).sum().float()

            epsilon = 1e-6  # small constant to avoid division by zero

            # Check if there are any positive samples.
            if num_pos.item() == 0 or num_neg.item() == 0:
                pos_weight_value = 1.0
            else:
                pos_weight_value = num_neg / (num_pos + epsilon)

            if isinstance(pos_weight_value, torch.Tensor):
                pos_weight_tensor = pos_weight_value.clone().detach().to(irmsd_confidence.device)
            else:
                pos_weight_tensor = torch.tensor(pos_weight_value, device=irmsd_confidence.device)

            # # Create a tensor for pos_weight_value on the correct device.
            # pos_weight_tensor = torch.tensor(pos_weight_value, device=irmsd_confidence.device)

            # Construct per-sample weights: use pos_weight_tensor for positive targets, 1 for negatives.
            weight = torch.where(irmsd_confidence == 1, pos_weight_tensor, torch.tensor(1.0, device=irmsd_confidence.device))

            print(f"weight: {weight}")
            print(f"irmsd_pred: {irmsd_pred}")
            print(f"irmsd_confidence: {irmsd_confidence}")

            # Compute the balanced BCE loss.
            # We use reduction='none' to get a per-element loss and then average over the desired dimensions.
            confidence_loss = F.binary_cross_entropy(irmsd_pred, irmsd_confidence, weight=weight, reduction='none').mean(dim=mean_dims)
            confidence_base_loss = F.binary_cross_entropy(torch.zeros_like(irmsd_pred), irmsd_confidence, weight=weight, reduction='none').mean(dim=mean_dims).detach()

            # # irmsd classification
            # irmsd_confidence = torch.cat([d.irmsd_confidence for d in data], dim=0) if isinstance(data, list) else data.irmsd_confidence #########
            # confidence_loss = F.binary_cross_entropy_with_logits(irmsd_pred, irmsd_confidence, reduction='none').mean(dim=mean_dims)
            # # confidence_base_loss is between all 0s and irmsd_confidence
            # confidence_base_loss = F.binary_cross_entropy_with_logits(torch.zeros_like(irmsd_pred), irmsd_confidence, reduction='none').mean(dim=mean_dims).detach()

            loss = lddt_loss * self.lddt_weight + confidence_loss * self.irmsd_confidence_weight
            base_loss = lddt_base_loss * self.lddt_weight + confidence_base_loss * self.irmsd_confidence_weight

            # print(f"confidence_loss: {confidence_loss}, confidence_base_loss: {confidence_base_loss}")

            l1_lambda = self.args.l1_lambda
            if l1_lambda > 0 and "l1_norm" in outputs:
                loss = loss + l1_lambda * outputs["l1_norm"]
            
            l2_lambda = self.args.l2_lambda
            if l2_lambda > 0 and "l2_norm" in outputs:
                loss = loss + l2_lambda * outputs["l2_norm"]
            

            return {
                "loss": loss,
                "base_loss": base_loss,
                "lddt_loss": lddt_loss,
                "lddt_base_loss": lddt_base_loss,
                "confidence_loss": confidence_loss,
                "confidence_base_loss": confidence_base_loss,
            }
        else:
            l1_lambda = self.args.l1_lambda
            if l1_lambda > 0 and "l1_norm" in outputs:
                loss = lddt_loss + l1_lambda * outputs["l1_norm"]
            else:
                loss = lddt_loss

            l2_lambda = self.args.l2_lambda
            if l2_lambda > 0 and "l2_norm" in outputs:
                loss = loss + l2_lambda * outputs["l2_norm"]
            
            return {
                "loss": loss,
                "base_loss": lddt_base_loss,
                "lddt_loss": lddt_loss,
                "lddt_base_loss": lddt_base_loss,
            }

        
    




def calc_complexRMSD_loss(pred_rec_coord, pred_lig_coord, true_rec_coord, true_lig_coord):
    # catenate the receptor and ligand coordinates
    pred_coord = torch.cat([pred_rec_coord, pred_lig_coord], dim=0)
    true_coord = torch.cat([true_rec_coord, true_lig_coord], dim=0)

    # kabsch alignment
    R, t = kabsch_torch(pred_coord.T, true_coord.T)
    aligned_pred_coord = pred_coord @ R.T + t.T

    # calculate the RMSD
    loss = torch.sqrt(torch.mean(torch.sum((aligned_pred_coord - true_coord)**2, dim=-1)))

    return loss

def calc_interface_rmsd_loss(pred_rec_coord, pred_lig_coord, true_rec_coord, true_lig_coord, loss_inter_cutoff=15.0):
    """
    Calculate the RMSD for the interface residues between the predicted and true coordinates.
    The interface is defined as residues of the receptor and ligand that are within the cutoff distance
    (e.g., 15 Å) in the true structure.
    
    Args:
    pred_rec_coord (torch.Tensor): Predicted coordinates of the receptor, shape (N_rec, 3).
    pred_lig_coord (torch.Tensor): Predicted coordinates of the ligand, shape (N_lig, 3).
    true_rec_coord (torch.Tensor): Ground truth coordinates of the receptor, shape (N_rec, 3).
    true_lig_coord (torch.Tensor): Ground truth coordinates of the ligand, shape (N_lig, 3).
    loss_inter_cutoff (float): Distance cutoff in angstroms to define interacting residues (default: 15.0).
    
    Returns:
    torch.Tensor: RMSD value for the interface residues.
    """
    # Calculate pairwise distances between true receptor and ligand coordinates
    rec_expanded = true_rec_coord.unsqueeze(1)  # Shape: (N_rec, 1, 3)
    lig_expanded = true_lig_coord.unsqueeze(0)  # Shape: (1, N_lig, 3)
    
    # Calculate squared distances for efficiency
    dist_sq = torch.sum((rec_expanded - lig_expanded) ** 2, dim=-1)  # Shape: (N_rec, N_lig)
    contact_mask = dist_sq <= (loss_inter_cutoff ** 2)  # Boolean mask for distances <= cutoff^2

    # Get unique indices of receptor and ligand residues that are in contact
    rec_indices = torch.unique(torch.nonzero(contact_mask, as_tuple=True)[0])
    lig_indices = torch.unique(torch.nonzero(contact_mask, as_tuple=True)[1])
    
    # If no residues are within the cutoff, return zero loss
    if rec_indices.numel() == 0 or lig_indices.numel() == 0:
        return torch.tensor(0.0, device=pred_rec_coord.device)

    # Gather the predicted and true coordinates for the interface residues
    pred_rec_interface = pred_rec_coord[rec_indices]  # Shape: (num_rec_interface, 3)
    pred_lig_interface = pred_lig_coord[lig_indices]  # Shape: (num_lig_interface, 3)
    true_rec_interface = true_rec_coord[rec_indices]  # Shape: (num_rec_interface, 3)
    true_lig_interface = true_lig_coord[lig_indices]  # Shape: (num_lig_interface, 3)

    # Concatenate the interface coordinates for RMSD calculation
    pred_interface_coords = torch.cat([pred_rec_interface, pred_lig_interface], dim=0)  # Shape: (num_interface, 3)
    true_interface_coords = torch.cat([true_rec_interface, true_lig_interface], dim=0)  # Shape: (num_interface, 3)


    R, t = kabsch_torch(pred_interface_coords.T, true_interface_coords.T)
    aligned_pred_coord = pred_interface_coords @ R.T + t.T

    # Calculate RMSD for the interface residues
    diff = aligned_pred_coord - true_interface_coords
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))

    return rmsd


class FAPEloss(nn.Module):
    """Frame aligned point error loss with per-residue weighting support.

    Args:
        Z (float, optional): Scaling factor for distances. Defaults to 10.0.
        clamp (float, optional): Maximum allowed distance before scaling. Defaults to 10.0.
        epsion (float, optional): Epsilon value (unused in this snippet). Defaults to -1e4.
    """
    def __init__(self, Z=10.0, clamp=10.0, epsion=-1e4):
        super().__init__()
        self.z = Z
        self.epsion = epsion
        self.clamp = clamp

    def forward(self, predict_bb_atoms, true_bb_atoms, pdb_mask=None, padding_mask=None, res_weights=None, device='cpu'):
        """
        Args:
            predict_bb_atoms (torch.Tensor): [batch, N_seq, 3, 3]
            true_bb_atoms (torch.Tensor): [batch, N_seq, 3, 3]
            pdb_mask (torch.Tensor, optional): pdb mask. Size: [batch, N_seq, N_seq]. Defaults to None.
            padding_mask (torch.Tensor, optional): padding mask. Size: [batch, N_seq, N_seq]. Defaults to None.
            res_weights (torch.Tensor, optional): Per-residue weights of shape (batch, N_seq). Defaults to None.
        """
        predict_R, predict_Trans = compute_residue_frames(predict_bb_atoms), predict_bb_atoms[:, :, 1, :] # Ca atoms
        RotationMatrix, translation = compute_residue_frames(true_bb_atoms), true_bb_atoms[:, :, 1, :] # Ca atoms

        # Compute pairwise translation differences for predicted and true translations
        delta_predict_Trans = rearrange(predict_Trans, 'b j t -> b j () t') - rearrange(predict_Trans, 'b i t -> b () i t')
        delta_Trans = rearrange(translation, 'b j t -> b j () t') - rearrange(translation, 'b i t -> b () i t')

        # Align points using the rotation matrices
        X_hat = torch.einsum('bikq, bjik->bijq', predict_R, delta_predict_Trans)
        X = torch.einsum('bikq, bjik->bijq', RotationMatrix, delta_Trans)

        # Compute pairwise distances and scale them
        distance = torch.norm(X_hat - X + 1e-8, dim=-1)
        distance = torch.clamp(distance, max=self.clamp) * (1/self.z)

        # Apply any provided masks
        if pdb_mask is not None:
            distance = distance * pdb_mask
        if padding_mask is not None:
            distance = distance * padding_mask

        # If per-residue weights are provided, expand to pairwise weights and compute a weighted average
        if res_weights is not None:
            # # Expand res_weights from shape (batch, N_seq) to (batch, N_seq, N_seq)
            # w_pair = res_weights.unsqueeze(2) * res_weights.unsqueeze(1)
            # # Multiply the distance by the pairwise weights
            # weighted_distance = distance * w_pair
            # # Normalize by the total weight sum to get a weighted average
            # total_weight = w_pair.sum()
            # FAPE_loss = weighted_distance.sum() / total_weight

            # distance: (batch, N_seq, N_seq), res_weights: (batch, N_seq)
            # Compute the mean distance for each residue (i.e. each row)
            row_avg = distance.mean(dim=2)  # shape: (batch, N_seq)
            # Compute the weighted average over rows, with weight for row i given by res_weights[i]
            # FAPE_loss = (res_weights * row_avg).sum() / (res_weights.sum() + 1e-6)
            FAPE_loss = (res_weights * row_avg).mean()
        else:
            # Default mean if no weights are provided
            FAPE_loss = torch.mean(distance)

        return FAPE_loss


def compute_residue_frames(res_coords):
    """
    Computes per-residue local reference frames (rotation matrices) from 
    residue coordinates.
    
    Args:
        res_coords (torch.Tensor): Tensor of shape (batch, res_n, 3, 3) where
                                   the 3 atoms are in the order [N, CA, C].
    
    Returns:
        torch.Tensor: Tensor of shape (batch, res_n, 3, 3) representing the 
                      rotation matrix for each residue. The columns of the 
                      rotation matrix are the x, y, and z axes.
    """
    # Extract coordinates: shape (batch, res_n, 3)
    N  = res_coords[:, :, 0, :]  # Nitrogen atom
    CA = res_coords[:, :, 1, :]  # Alpha Carbon atom
    C  = res_coords[:, :, 2, :]  # Carbon atom

    # x-axis: normalized vector from CA to C
    x = C - CA
    x = x / x.norm(dim=-1, keepdim=True)
    
    # Temporary vector: from CA to N
    v = N - CA
    
    # z-axis: cross product of x and v, normalized.
    z = torch.cross(x, v, dim=-1)
    z = z / z.norm(dim=-1, keepdim=True)
    
    # y-axis: cross product of z and x to complete a right-handed system
    y = torch.cross(z, x, dim=-1)
    
    # Construct the rotation matrix by stacking the axes as columns.
    # The resulting matrix has shape (batch, res_n, 3, 3)
    frame = torch.stack([x, y, z], dim=-1)
    
    return frame





