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
class flexFM_RtLoss(nn.Module):
    def __init__(self, args):
        super(flexFM_RtLoss, self).__init__()
        self.args = args
        self.flex_rot_weight = args.flex_rot_weight # 1
        self.flex_tr_weight = args.flex_tr_weight # 1
        self.tr_weight = args.tr_weight # 1
        self.rot_weight = args.rot_weight # 1
        self.lddt_weight = args.lddt_weight # 1
        self.flex_fape_weight = args.flex_fape_weight # 1
        
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
        
        coord_r_pred = outputs["coord_r_pred"]
        coord_l_pred = outputs["coord_l_pred"]
        lddt_pred = outputs["lddt_pred"]
        rot_pred = outputs["rot_pred"]
        tr_pred = outputs["tr_pred"]

        if torch.isnan(coord_r_pred).any():
            print("NaN detected in output coord_r_pred!")
        if torch.isnan(coord_l_pred).any():
            print("NaN detected in output coord_l_pred!")
        if torch.isnan(lddt_pred).any():
            print("NaN detected in output lddt_pred!")
        if torch.isnan(rot_pred).any():
            print("NaN detected in output rot_pred!")
        if torch.isnan(tr_pred).any():
            print("NaN detected in output tr_pred!")


        if self.args.res_rot_flex:
            rec_res_rot_pred = outputs["rec_res_rot_pred"]
            lig_res_rot_pred = outputs["lig_res_rot_pred"]
            if torch.isnan(rec_res_rot_pred).any():
                print("NaN detected in output rec_res_rot_pred!")
            if torch.isnan(lig_res_rot_pred).any():
                print("NaN detected in output lig_res_rot_pred!")

        device = coord_r_pred.device

        if not isinstance(data, list):
            data = data.to_data_list()

        # sigmas
        data_t = [torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type] for noise_type in ["flex", "flex", "rot", "tr"]]
        flex_rec_s, flex_lig_s, rot_s, tr_s = NoiseSchedule_flexFM_Rt(self.args)(*data_t)

        # lddt
        lddt = torch.cat([d.lddt for d in data], dim=0) if device.type == 'cuda' else data.lddt

        lddt_loss = ((lddt_pred - lddt) ** 2).mean(dim=mean_dims)
        lddt_base_loss = (lddt ** 2).mean(dim=mean_dims).detach()

        # print(f"lddt_loss: {lddt_loss}, lddt_base_loss: {lddt_base_loss}")

        
        # residue translation component
        rec_res_decay_weight = torch.cat([d['receptor'].res_decay_weight for d in data], dim=0) if device.type == 'cuda' else data['receptor'].res_decay_weight
        lig_res_decay_weight = torch.cat([d['ligand'].res_decay_weight for d in data], dim=0) if device.type == 'cuda' else data['ligand'].res_decay_weight

        # print(f"rec_res_decay_weight1: {rec_res_decay_weight}, lig_res_decay_weight1: {lig_res_decay_weight}")

        rec_res_gap_masks = torch.zeros_like(rec_res_decay_weight) # if device.type == 'cuda' else data.gap_masks
        lig_res_gap_masks = torch.zeros_like(lig_res_decay_weight) # if device.type == 'cuda' else data.gap_masks

        # res_loss_weight = torch.cat([d.res_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.res_loss_weight
        
        rec_res_decay_weight = rec_res_decay_weight * (1.-rec_res_gap_masks)
        rec_res_decay_weight = rec_res_decay_weight / rec_res_decay_weight.sum() * (1.-rec_res_gap_masks).sum()
        lig_res_decay_weight = lig_res_decay_weight * (1.-lig_res_gap_masks)
        lig_res_decay_weight = lig_res_decay_weight / lig_res_decay_weight.sum() * (1.-lig_res_gap_masks).sum()

        # print(f"rec_res_decay_weight2: {rec_res_decay_weight}, lig_res_decay_weight2: {lig_res_decay_weight}")

        rec_res_loss_weight = torch.cat([d['receptor'].res_loss_weight for d in data], dim=0) if device.type == 'cuda' else data['receptor'].res_loss_weight
        # print(f"rec_res_loss_weight: {rec_res_loss_weight}")
        rec_res_loss_weight = rec_res_decay_weight  * rec_res_loss_weight
        lig_res_loss_weight = torch.cat([d['ligand'].res_loss_weight for d in data], dim=0) if device.type == 'cuda' else data['ligand'].res_loss_weight
        # print(f"lig_res_loss_weight: {lig_res_loss_weight}")
        lig_res_loss_weight = lig_res_decay_weight  * lig_res_loss_weight


        # local flexing component (translation)
        tr_rec_score = torch.cat([d.flex_rec_score for d in data], dim=0) if device.type == 'cuda' else data.tr_rec_score
        # rec_tr_loss = torch.nn.L1Loss(reduction='none')(coord_r_pred.cpu(),tr_rec_score).mean(dim=1) * rec_res_loss_weight.squeeze(1) * 3.0#((res_tr_pred.cpu() - res_tr_score) ** 2).mean(dim=mean_dims)
        rec_tr_loss = torch.nn.L1Loss(reduction='none')(coord_r_pred,tr_rec_score).mean(dim=1) * rec_res_loss_weight.squeeze(1) * 3.0#((res_tr_pred.cpu() - res_tr_score) ** 2).mean(dim=mean_dims)
        rec_tr_base_loss = (tr_rec_score).abs().mean(dim=1).detach() * rec_res_loss_weight.squeeze(1) * 3.0

        tr_lig_score = torch.cat([d.flex_lig_score for d in data], dim=0) if device.type == 'cuda' else data.tr_lig_score
        # lig_tr_loss = torch.nn.L1Loss(reduction='none')(coord_l_pred.cpu(),tr_lig_score).mean(dim=1) * lig_res_loss_weight.squeeze(1) * 3.0#((res_tr_pred.cpu() - res_tr_score) ** 2).mean(dim=mean_dims)
        lig_tr_loss = torch.nn.L1Loss(reduction='none')(coord_l_pred,tr_lig_score).mean(dim=1) * lig_res_loss_weight.squeeze(1) * 3.0#((res_tr_pred.cpu() - res_tr_score) ** 2).mean(dim=mean_dims)
        lig_tr_base_loss = (tr_lig_score).abs().mean(dim=1).detach() * lig_res_loss_weight.squeeze(1) * 3.0

        # local flexing component (rotation)
        if self.args.res_rot_flex:
            rec_res_rot_score = torch.cat([d.flex_rec_rot_score for d in data], dim=0) if device.type == 'cuda' else data.flex_rec_rot_score

            # res_rot_pred_norm = res_rot_pred.norm(dim=-1,keepdim=True).cpu()
            # res_rot_pred_vec = res_rot_pred.cpu() / (res_rot_pred_norm+1e-12)

            rec_res_rot_loss_pos = (torch.nn.L1Loss(reduction='none')(rec_res_rot_pred,rec_res_rot_score)).mean(dim=1)
            # res_rot_loss_neg = (torch.nn.L1Loss(reduction='none')((res_rot_pred_norm-2*np.pi)*res_rot_pred_vec,res_rot_score)).mean(dim=1)
            # res_rot_loss = torch.minimum(res_rot_loss_pos,res_rot_loss_neg)
            rec_res_rot_loss = rec_res_rot_loss_pos * rec_res_loss_weight.squeeze(1) * 15.0
            rec_res_rot_base_loss = (rec_res_rot_score.abs()).mean(dim=1).detach() * rec_res_loss_weight.squeeze(1) * 15.0

            lig_res_rot_score = torch.cat([d.flex_lig_rot_score for d in data], dim=0) if device.type == 'cuda' else data.flex_lig_rot_score
            lig_res_rot_loss_pos = (torch.nn.L1Loss(reduction='none')(lig_res_rot_pred,lig_res_rot_score)).mean(dim=1)
            lig_res_rot_loss = lig_res_rot_loss_pos * lig_res_loss_weight.squeeze(1) * 15.0
            lig_res_rot_base_loss = (lig_res_rot_score.abs()).mean(dim=1).detach() * lig_res_loss_weight.squeeze(1) * 15.0


            if apply_mean:
                rec_res_rot_loss = rec_res_rot_loss.mean()
                rec_res_rot_base_loss = rec_res_rot_base_loss.mean()
                lig_res_rot_loss = lig_res_rot_loss.mean()
                lig_res_rot_base_loss = lig_res_rot_base_loss.mean()

            if self.args.FAPE_loss:

                # separate the batch data
                rec_batch = batch_data['receptor'].batch
                lig_batch = batch_data['ligand'].batch

                # Find common unique graph indices between receptor and ligand graphs
                common_graphs = rec_batch.unique()[torch.isin(rec_batch.unique(), lig_batch.unique())]

                fapeloss = FAPEloss()

                fape_rec_loss = []
                fape_lig_loss = []
                fape_rec_base_loss = []
                fape_lig_base_loss = []

                for graph_idx in common_graphs:
     
                    # Receptor data for the current graph
                    # [n, 3], [n, 3, 3], [n, 3, 3], [n, 3, 3]
                    delta_r, rot_r, bb_atoms_r_cur, bb_atoms_r_gt = coord_r_pred[rec_batch == graph_idx], axis_angle_to_matrix(rec_res_rot_pred[rec_batch == graph_idx]), batch_data["receptor"].bb_atoms[rec_batch == graph_idx], batch_data["receptor"].bb_atoms_b[rec_batch == graph_idx]
                    
                    # Ligand data for the current graph
                    # [n, 3], [n, 3, 3], [n, 3, 3], [n, 3, 3]
                    delta_l, rot_l, bb_atoms_l_cur, bb_atoms_l_gt = coord_l_pred[lig_batch == graph_idx], axis_angle_to_matrix(lig_res_rot_pred[lig_batch == graph_idx]), batch_data["ligand"].bb_atoms[lig_batch == graph_idx], batch_data["ligand"].bb_atoms_b[lig_batch == graph_idx]

                    # [n, 3, 3]
                    bb_atoms_l_pred = (bb_atoms_l_cur - bb_atoms_l_cur[:,[1],:]) @ rot_l.transpose(1,2) + bb_atoms_l_cur[:,[1],:] + delta_l[:,None,:]
                    bb_atoms_r_pred = (bb_atoms_r_cur - bb_atoms_r_cur[:,[1],:]) @ rot_r.transpose(1,2) + bb_atoms_r_cur[:,[1],:] + delta_r[:,None,:]

                    # [n, 3, 3]
                    bb_atoms_l_base = bb_atoms_l_cur
                    bb_atoms_r_base = bb_atoms_r_cur

                    # lig_res_loss_weight/rec_res_loss_weight to shape (1, -1)
                    lig_res_loss_weight_reshaped = lig_res_loss_weight[lig_batch == graph_idx].view(1, -1)
                    rec_res_loss_weight_reshaped = rec_res_loss_weight[rec_batch == graph_idx].view(1, -1)

                    # print("lig_res_loss_weight shape", lig_res_loss_weight.shape)
                    # print("rec_res_loss_weight shape", rec_res_loss_weight.shape)

                    # predict_bb_atoms, true_bb_atoms, pdb_mask=None, padding_mask=None, res_weights=None
                    fape_rec_loss.append(fapeloss(bb_atoms_r_pred.unsqueeze(0), bb_atoms_r_gt.unsqueeze(0), res_weights=rec_res_loss_weight_reshaped))
                    fape_rec_base_loss.append(fapeloss(bb_atoms_r_base.unsqueeze(0), bb_atoms_r_gt.unsqueeze(0), res_weights=rec_res_loss_weight_reshaped))

                    fape_lig_loss.append(fapeloss(bb_atoms_l_pred.unsqueeze(0), bb_atoms_l_gt.unsqueeze(0), res_weights=lig_res_loss_weight_reshaped))
                    fape_lig_base_loss.append(fapeloss(bb_atoms_l_base.unsqueeze(0), bb_atoms_l_gt.unsqueeze(0), res_weights=lig_res_loss_weight_reshaped))

                fape_rec_loss = torch.stack(fape_rec_loss).mean()
                fape_lig_loss = torch.stack(fape_lig_loss).mean()

                fape_rec_base_loss = torch.stack(fape_rec_base_loss).mean()
                fape_lig_base_loss = torch.stack(fape_lig_base_loss).mean()

                # print(f"fape_rec_loss: {fape_rec_loss}, fape_lig_loss: {fape_lig_loss}")
                # print(f"fape_rec_base_loss: {fape_rec_base_loss}, fape_lig_base_loss: {fape_lig_base_loss}")
                

        
        # global lig translation component
        tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
        tr_s = tr_s.unsqueeze(-1)
        tr_loss = ((tr_pred - tr_score) ** 2 / tr_s ** 2).mean(dim=mean_dims)
        tr_base_loss = (tr_score ** 2 / tr_s ** 2).mean(dim=mean_dims).detach()

        # global lig rotation component
        # rot_loss_weight = torch.cat([d.rot_loss_weight for d in data], dim=0) if device.type == 'cuda' else data.rot_loss_weight
        rot_score = torch.cat([d.rot_score.unsqueeze(0) for d in data], dim=0) if device.type == 'cuda' else data.rot_score
        rot_pred_norm = rot_pred.norm(dim=-1,keepdim=True)
        rot_pred_vec = rot_pred / (rot_pred_norm+1e-6)
        # print(rot_pred.shape)
        # print(rot_score.shape)
        rot_loss_pos = (((rot_pred - rot_score) / rot_s[...,None]) ** 2).mean(dim=1)
        rot_loss_neg = ((((rot_pred_norm-2*np.pi)*rot_pred_vec - rot_score) / rot_s[...,None]) ** 2).mean(dim=1)
        # print(rot_loss_pos, rot_loss_neg)

        rot_loss = torch.minimum(rot_loss_pos,rot_loss_neg)
        # print("rot_loss", rot_loss)
        # print("rot_base_loss", ((rot_score / rot_s[...,None]) ** 2).mean(dim=1))

        rot_base_loss = ((rot_score / rot_s[...,None]) ** 2).mean(dim=mean_dims).detach()
        # print(rot_base_loss)

        if apply_mean: # default as True
            rot_loss = rot_loss.mean()

        # # DiffDock-PP
        # rot_score = (
        #     torch.cat([d.rot_score.unsqueeze(0) for d in data], dim=0)
        #     if device.type == "cuda" 
        #     else data.rot_score
        # )
        # rot_score_norm = score_norm(rot_s).unsqueeze(-1)
        # rot_loss = (((rot_pred - rot_score) /
        #     (rot_score_norm + self.eps)) ** 2).mean(
        #     dim=mean_dims
        # )
        # rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()



        if apply_mean: # default as True
            rec_tr_loss = rec_tr_loss.mean()
            rec_tr_base_loss = rec_tr_base_loss.mean()
            lig_tr_loss = lig_tr_loss.mean()
            lig_tr_base_loss = lig_tr_base_loss.mean()

        if self.args.lateFlex and self.args.epoch < self.args.lateFlex_epoch:
            self.flex_tr_weight = 0.0
            self.flex_rot_weight = 0.0


        if self.args.res_rot_flex:
            if not self.args.FAPE_loss:
                loss = lddt_loss * self.lddt_weight + rec_tr_loss*self.flex_tr_weight + lig_tr_loss*self.flex_tr_weight + rot_loss*self.rot_weight + tr_loss*self.tr_weight + \
                    rec_res_rot_loss * self.flex_rot_weight + lig_res_rot_loss * self.flex_rot_weight
                base_loss = lddt_base_loss * self.lddt_weight + rec_tr_base_loss*self.flex_tr_weight + lig_tr_base_loss*self.flex_tr_weight + rot_base_loss*self.rot_weight + tr_base_loss*self.tr_weight + \
                    rec_res_rot_base_loss * self.flex_rot_weight + lig_res_rot_base_loss * self.flex_rot_weight
            else:
                loss = lddt_loss * self.lddt_weight + rec_tr_loss*self.flex_tr_weight + lig_tr_loss*self.flex_tr_weight + rot_loss*self.rot_weight + tr_loss*self.tr_weight + \
                    rec_res_rot_loss * self.flex_rot_weight + lig_res_rot_loss * self.flex_rot_weight + fape_rec_loss*self.flex_fape_weight + fape_lig_loss*self.flex_fape_weight
                base_loss = lddt_base_loss * self.lddt_weight + rec_tr_base_loss*self.flex_tr_weight + lig_tr_base_loss*self.flex_tr_weight + rot_base_loss*self.rot_weight + tr_base_loss*self.tr_weight + \
                    rec_res_rot_base_loss * self.flex_rot_weight + lig_res_rot_base_loss * self.flex_rot_weight + fape_rec_base_loss*self.flex_fape_weight + fape_lig_base_loss*self.flex_fape_weight
        else:
            loss = lddt_loss * self.lddt_weight + rec_tr_loss*self.flex_tr_weight + lig_tr_loss*self.flex_tr_weight + rot_loss*self.rot_weight + tr_loss*self.tr_weight
            base_loss = lddt_base_loss * self.lddt_weight + rec_tr_base_loss*self.flex_tr_weight + lig_tr_base_loss*self.flex_tr_weight + rot_base_loss*self.rot_weight + tr_base_loss*self.tr_weight




        # [dia]
        print(f"##### Loss: {loss}, data {batch_data.name} #####")

        if self.args.res_rot_flex:
            if not self.args.FAPE_loss:
                losses = {
                    "loss": loss,
                    "base_loss": base_loss,
                    "lddt_loss": lddt_loss,
                    "lddt_base_loss": lddt_base_loss,
                    "rec_flex_loss": rec_tr_loss,
                    "lig_flex_loss": lig_tr_loss,
                    "rec_flex_base_loss": rec_tr_base_loss,
                    "lig_flex_base_loss": lig_tr_base_loss,
                    "rot_loss": rot_loss,
                    "rot_base_loss": rot_base_loss,
                    "tr_loss": tr_loss,
                    "tr_base_loss": tr_base_loss,
                    "rec_res_rot_loss": rec_res_rot_loss,
                    "lig_res_rot_loss": lig_res_rot_loss,
                    "rec_res_rot_base_loss": rec_res_rot_base_loss,
                    "lig_res_rot_base_loss": lig_res_rot_base_loss,
                }
            else:
                losses = {
                    "loss": loss,
                    "base_loss": base_loss,
                    "lddt_loss": lddt_loss,
                    "lddt_base_loss": lddt_base_loss,
                    "rec_flex_loss": rec_tr_loss,
                    "lig_flex_loss": lig_tr_loss,
                    "rec_flex_base_loss": rec_tr_base_loss,
                    "lig_flex_base_loss": lig_tr_base_loss,
                    "rot_loss": rot_loss,
                    "rot_base_loss": rot_base_loss,
                    "tr_loss": tr_loss,
                    "tr_base_loss": tr_base_loss,
                    "rec_res_rot_loss": rec_res_rot_loss,
                    "lig_res_rot_loss": lig_res_rot_loss,
                    "rec_res_rot_base_loss": rec_res_rot_base_loss,
                    "lig_res_rot_base_loss": lig_res_rot_base_loss,
                    "fape_rec_loss": fape_rec_loss,
                    "fape_lig_loss": fape_lig_loss,
                    "fape_rec_base_loss": fape_rec_base_loss,
                    "fape_lig_base_loss": fape_lig_base_loss,
                }
        else:
            losses = {
                "loss": loss,
                "base_loss": base_loss,
                "lddt_loss": lddt_loss,
                "lddt_base_loss": lddt_base_loss,
                "rec_flex_loss": rec_tr_loss,
                "lig_flex_loss": lig_tr_loss,
                "rec_flex_base_loss": rec_tr_base_loss,
                "lig_flex_base_loss": lig_tr_base_loss,
                "rot_loss": rot_loss,
                "rot_base_loss": rot_base_loss,
                "tr_loss": tr_loss,
                "tr_base_loss": tr_base_loss,
            }

        # print(f"##### losses: {losses}, data {batch_data.name} #####")

        return losses
    




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





