"""
This is modified according to the nm coeffs diffusion

2023.11.25
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
import math

from .so3 import sample_vec, score_vec
from .torus import score as score_torus
from .geometry import kabsch_torch, axis_angle_to_matrix

import prody

EPS = 1e-6


# check if anywhere the functions are used !!!!

# ------ PyG UTILS -------
class NoiseTransform_flexSDE_Rt(BaseTransform):
    """
        Apply translation, rotation, nm noise
    """
    def __init__(self, args):
        # save min/max sigma scales
        self.noise_schedule = NoiseSchedule_flexSDE_Rt(args)
        self.args = args

    def __call__(self, data):
        """
            Modifies data in place
            @param (torch_geometric.data.HeteroData) data
        """
        t = np.random.uniform()  # sample time
        t_flex_rec, t_flex_lig, t_rot, t_tr = t, t, t, t  # same time scale for each
        data = self.apply_noise(data, t_flex_rec, t_flex_lig, t_rot, t_tr)
        return data

    def apply_noise(self, data, t_flex_rec, t_flex_lig, t_rot, t_tr,
                    flex_rec_update=None,
                    flex_lig_update=None,
                    rot_update=None,
                    tr_update=None):
        """
            Apply noise to existing HeteroData object
            @param (torch_geometric.data.HeteroData) data
        """
        # T_rec = 0.4/data["receptor"].smallest_lambda if "smallest_lambda" in data["receptor"] else None
        # T_lig = 0.4/data["ligand"].smallest_lambda if "smallest_lambda" in data["ligand"] else None
        flex_tr_rec_s, flex_tr_lig_s, flex_rec_s, flex_lig_s, rot_s, tr_s = self.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)

        training_flag = False

        if flex_rec_update is None: # in training
            training_flag = True

        # set the num_nodes according to pos_b_ori
        data["receptor"].num_nodes = data["receptor"].pos_b_ori.shape[0]
        data["ligand"].num_nodes = data["ligand"].pos_b_ori.shape[0]
            
        set_time_flexSDE_Rt(data, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device=None)

        # sample updates if not provided (in training)
        if training_flag:

            # sample rot and tr update
            tr_update = torch.normal(mean=0, std=tr_s, size=(1, 3))
            
            rot_update = sample_vec(eps=rot_s)
            rot_update = torch.from_numpy(rot_update).float()

            # tr_update_norm
            tr_update_norm = tr_update.norm(dim=-1).item()


            orig_ca_lig_cross_distances = (data['ligand'].pos_b_ori[None,...] - data['receptor'].pos_b_ori[:,None,...]).norm(dim=-1)
            # print(orig_ca_lig_cross_distances.shape)


            # res_decay_weight: residues to nearest residues in another protein/residues to another protein center distance decay weight
            if self.args.nearest_res_decay:
                rec_distance=torch.min(orig_ca_lig_cross_distances, dim=-1).values
                lig_distance=torch.min(orig_ca_lig_cross_distances, dim=0).values
                data['receptor'].res_decay_weight = torch.exp(-torch.nn.ReLU()((rec_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))
                data['ligand'].res_decay_weight = torch.exp(-torch.nn.ReLU()((lig_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))
            else:
                lig_center = data['ligand'].pos_b_ori.mean(dim=0,keepdim=True)
                rec_center = data['receptor'].pos_b_ori.mean(dim=0,keepdim=True)
                lig_distance = (data['ligand'].pos_b_ori-rec_center).norm(dim=-1,keepdim=True)
                rec_distance = (data['receptor'].pos_b_ori-lig_center).norm(dim=-1,keepdim=True)
                data['receptor'].res_decay_weight = torch.exp(-torch.nn.ReLU()((rec_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))
                data['ligand'].res_decay_weight = torch.exp(-torch.nn.ReLU()((lig_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))


            

            # pos_bb_atoms_u_lig = data['ligand'].bb_atoms_u[data["ligand"].mask]

            # new_pos_bb_atoms_u_lig = (
            #     (pos_bb_atoms_u_lig.reshape((-1, 3)) - com) @ rot_mat.T + lig_rand_tr + com
            # )
            # new_pos_bb_atoms_u_lig = new_pos_bb_atoms_u_lig.reshape((-1, 3, 3))

            # # # bb_atoms
            # # data['ligand'].bb_atoms_u_t = item[f"{key}_u_bb_coords"].float()
            

            # v_t
            # rec_sigma = torch.clamp(flex_rec_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise
            # lig_sigma = torch.clamp(flex_lig_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise 
            rec_sigma = torch.clamp(flex_rec_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise
            lig_sigma = torch.clamp(flex_lig_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise 


            # current res_loss_weight
            data['receptor'].res_loss_weight = torch.ones(data['receptor'].pos_b_ori.shape[0],1).float() * (tr_update_norm<self.args.zeroing_cutoff) # torch.ones(data['receptor'].pos.shape[0],1).float() * (tr_update_norm<6.)
            data['ligand'].res_loss_weight = torch.ones(data['ligand'].pos_b_ori.shape[0],1).float() * (tr_update_norm<self.args.zeroing_cutoff) # torch.ones(data['receptor'].pos.shape[0],1).float() * (tr_update_norm<6.)

            # flexing updates
            sde = SDE(data, self.args)
            flex_rec_update, flex_lig_update, flex_rec_score, flex_lig_score, flex_rec_score_norm, flex_lig_score_norm = sde.get_train_updates(flex_tr_rec_s, flex_tr_lig_s)
            # # u(t) tr_rec*
            # flex_rec_update = (data['receptor'].pos_u[data['receptor'].mask]-data['receptor'].pos_b_ori) * rec_sigma  # + torch.normal(mean=0, std=0.3, size=(data['receptor'].pos_b_ori.shape[0], 3))
            # # u(t) tr_lig* 
            # flex_lig_update = (data['ligand'].pos_u[data['ligand'].mask]-data['ligand'].pos_b_ori) * lig_sigma  # + torch.normal(mean=0, std=0.3, size=(data['ligand'].pos_b_ori.shape[0], 3))
            
            if self.args.res_rot_flex:
                # u(t) rot_rec*
                flex_rec_rot_update = (data['receptor'].rotvecs) * rec_sigma  # + torch.normal(mean=0, std=0.3, size=(data['receptor'].pos_b_ori.shape[0], 3))
                # u(t) rot_lig* 
                flex_lig_rot_update = (data['ligand'].rotvecs) * lig_sigma  # + torch.normal(mean=0, std=0.3, size=(data['ligand'].pos_b_ori.shape[0], 3))

                data['receptor'].bb_atoms = data['receptor'].bb_atoms_b
                data['ligand'].bb_atoms = data['ligand'].bb_atoms_b
                

    
            data["receptor"].pos = data["receptor"].pos_b_ori
            data["ligand"].pos = data["ligand"].pos_b_ori
             
            # initialize edge_index and x
            # initialize node features as the unbound ones in training
            if 'x_ori' in data["receptor"]:
                data["receptor"].x = data["receptor"].x_ori
                data["ligand"].x = data["ligand"].x_ori
            

                # initialize edge index as the unbound ones in training
                data["receptor", "contact", "receptor"].edge_index = data["receptor"].edge_index_b
                data["ligand", "contact", "ligand"].edge_index = data["ligand"].edge_index_b
            else:
                raise ValueError("No x_ori in data")
                

        if self.args.res_rot_flex:
            # apply updates
            self.apply_updates(data, flex_rec_update=flex_rec_update,
                            flex_lig_update=flex_lig_update,
                            rot_update=rot_update,
                            tr_update=tr_update,
                            flex_rec_rot_update=flex_rec_rot_update,
                            flex_lig_rot_update=flex_lig_rot_update)

            # compute ground truth score given updates, noise level
            self.get_score(data, flex_rec_score, flex_lig_score, flex_rec_score_norm, flex_lig_score_norm,
                        rot_update, tr_update, flex_rec_rot_update=flex_rec_rot_update,
                        flex_lig_rot_update=flex_lig_rot_update, orig_ca_lig_cross_distances=orig_ca_lig_cross_distances)

        else:
            # apply updates
            self.apply_updates(data, flex_rec_update=flex_rec_update,
                            flex_lig_update=flex_lig_update,
                            rot_update=rot_update,
                            tr_update=tr_update)

            # compute ground truth score given updates, noise level
            self.get_score(data, flex_rec_score, flex_lig_score, flex_rec_score_norm, flex_lig_score_norm,
                        rot_update, tr_update, orig_ca_lig_cross_distances=orig_ca_lig_cross_distances)

        return data

    def apply_updates(self, data,
                      flex_rec_update=None,flex_lig_update=None,rot_update=None,tr_update=None, flex_rec_rot_update=None, flex_lig_rot_update=None):
        """
            Apply translation / rotation to
            data["ligand"].pos (shape: (n_res, 3)) and data["receptor"].pos (shape: (n_res, 3))
            @param (torch_geometric.data.HeteroData) data
        """

        flex_new_pos_l = data["ligand"].pos
        flex_new_pos_r = data["receptor"].pos

        #
        # lig Ca pos: flex + tr + rot
        new_pos_l = (
            flex_new_pos_l + flex_lig_update
        )
        # # kabsch to separate the residue-level translation and the global rigid-body motions
        # R, t = kabsch_torch(new_pos_l.T, flex_new_pos_l.T)
        # new_pos_l = new_pos_l @ R.T + t.T
        # # [print_]
        # print("kabsch R, t: ", R, t)
        # flex_lig_update = new_pos_l - flex_new_pos_l

        com = torch.mean(new_pos_l, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())

        new_pos_l = (new_pos_l -com) @ rot_mat.T + com + tr_update

        #
        # rec Ca pos: flex
        new_pos_r = (
            flex_new_pos_r + flex_rec_update
        )
        # # kabsch to separate the residue-level translation and the global rigid-body motions
        # R, t = kabsch_torch(new_pos_r.T, flex_new_pos_r.T)
        # new_pos_r = new_pos_r @ R.T + t.T
        # flex_rec_update = new_pos_r - flex_new_pos_r

        
        data["ligand"].pos = new_pos_l
        data["receptor"].pos = new_pos_r

        if "pos_u_Rt" in data["receptor"]:
            data["receptor"].pos_u_Rt = data["receptor"].pos_u_Rt
            pos_u_Rt_lig = data["ligand"].pos_u_Rt
            com_pos_u_Rt_lig = torch.mean(pos_u_Rt_lig, dim=0, keepdim=True)
            new_pos_u_Rt_lig = (pos_u_Rt_lig - com_pos_u_Rt_lig) @ rot_mat.T + com_pos_u_Rt_lig + tr_update
            data["ligand"].pos_u_Rt = new_pos_u_Rt_lig

        # rec bb_atoms: (flex_rot + flex_tr)
        # lig bb_atoms: (flex_rot + flex_tr) + (rot + tr)
        if flex_rec_rot_update is not None:
            rec_res_rot_mat = axis_angle_to_matrix(flex_rec_rot_update)
            lig_res_rot_mat = axis_angle_to_matrix(flex_lig_rot_update)
            # rec bb_atoms: (flex_rot + flex_tr)
            data['receptor'].bb_atoms = (data['receptor'].bb_atoms - data['receptor'].bb_atoms[:,[1],:]) @ rec_res_rot_mat.transpose(1,2) + data['receptor'].bb_atoms[:,[1],:] + flex_rec_update[:,None,:]
            # lig bb_atoms: (flex_rot + flex_tr) + (rot + tr)
            flex_lig_bb_atoms = (data['ligand'].bb_atoms - data['ligand'].bb_atoms[:,[1],:]) @ lig_res_rot_mat.transpose(1,2) + data['ligand'].bb_atoms[:,[1],:] + flex_lig_update[:,None,:]
            # print(f"flex_lig_bb_atoms shape: {flex_lig_bb_atoms.shape}")
            # print(f"com shape: {com.shape}")
            # print(f"rot_mat shape: {rot_mat.shape}")
            # print(f"tr_update shape: {tr_update.shape}")
            data['ligand'].bb_atoms = (flex_lig_bb_atoms - com[:,None,:]) @ rot_mat[None,:,:].transpose(1,2) + com[:,None,:] + tr_update[:,None,:] # still use Ca center

        return data

    def get_score(self, data, flex_rec_score, flex_lig_score, flex_rec_score_norm, flex_lig_score_norm,
                  rot_update, tr_update, flex_rec_rot_update=None,
                  flex_lig_rot_update=None,orig_ca_lig_cross_distances=None):
        """
            Compute ground truth score, given updates and noise.
            Modifies data in place.
        """
        data.flex_rec_score_norm = flex_rec_score_norm
        data.flex_lig_score_norm = flex_lig_score_norm

        # rec flex: -u(t) tr_rec*
        data.flex_rec_score = flex_rec_score
        # lig flex: -u(t) tr_lig* @ R_t
        # com = torch.mean(flex_lig_update, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())
        # flex_lig_update = (flex_lig_update -com) @ rot_mat.T + com
        flex_lig_score = flex_lig_score @ rot_mat.T
        data.flex_lig_score = flex_lig_score

        #[print_]
        # print("flex_rec_score", torch.norm(flex_rec_score, dim=-1).mean(), "flex_lig_score", torch.norm(flex_lig_score, dim=-1).mean())

        if flex_rec_rot_update is not None:
            # rec frame rot flex: -u(t) rot_rec*
            data.flex_rec_rot_score = -flex_rec_rot_update
            # lig frame rot flex: (R_t^-1 @ u(t) rot_lig* @ R_t)^-1
            lig_res_rot_mat = axis_angle_to_matrix(flex_lig_rot_update)
            overall_lig_rot_mat = rot_mat[None,:,:] @ lig_res_rot_mat.transpose(1,2) @ rot_mat[None,:,:].transpose(1,2) # (n,3,3)
            data.flex_lig_rot_score = rotation_matrices_to_vectors(overall_lig_rot_mat)


        data.tr_score = -tr_update
        data.rot_score = -rot_update

        # clddt
        if orig_ca_lig_cross_distances is not None:
            ca_lig_cross_distances = (data['ligand'].pos[None,...] - data['receptor'].pos[:,None,...]).norm(dim=-1)
            ca_lig_cross_distances_diff = (orig_ca_lig_cross_distances - ca_lig_cross_distances).abs()

            cutoff_mask = (orig_ca_lig_cross_distances < 15.0).float()
            score = 0.25 * ( (ca_lig_cross_distances_diff<1.0).float() # 0.5
                            + (ca_lig_cross_distances_diff<2.0).float() # 1.0
                            + (ca_lig_cross_distances_diff<4.0).float() # 2.0
                            + (ca_lig_cross_distances_diff<8.0).float()) # 4.0
            
            if cutoff_mask.sum() > 0:
                ca_lig_cross_lddt = (score * cutoff_mask).sum() / cutoff_mask.sum()
            else:
                print(f"{data.name} has no valid ca_lig_cross_distances under 15.0")
                ca_lig_cross_lddt = torch.tensor(0.0)  

            ca_lig_cross_lddt =  (score * cutoff_mask).sum() / cutoff_mask.sum()
            data.lddt = ca_lig_cross_lddt[None,None]

        return data


class NoiseSchedule_flexSDE_Rt:
    """
        Transforms t into scaled sigmas
    """
    def __init__(self, args):
        self.args = args
        self.flex_rec_s_min = args.flex_rec_s_min
        self.flex_rec_s_max = args.flex_rec_s_max
        self.flex_lig_s_max = args.flex_lig_s_max
        self.flex_lig_s_min = args.flex_lig_s_min
        self.tr_s_min = args.tr_s_min
        self.tr_s_max = args.tr_s_max
        self.rot_s_max = args.rot_s_max
        self.rot_s_min = args.rot_s_min

        # 
        self.A_factor = args.A_factor
        self.B_exp = args.B_exp

        #
        self.c = args.c

        #
        self.q = args.sde_tSchedule_factor


    def __call__(self, t_flex_rec, t_flex_lig, t_rot, t_tr, T=5.0):
        """
            Convert from time to (scaled) sigma space
        """
        tr_s = self.tr_s_min ** (1 - t_tr) * self.tr_s_max**t_tr
        rot_s = self.rot_s_min ** (1 - t_rot) * self.rot_s_max**t_rot

        if torch.is_tensor(t_flex_rec):

            if self.args.flex_tr_t_schedule == "log":
                flex_tr_rec_s = (-self.q*torch.log1p(-t_flex_rec)+self.flex_rec_s_min).float().to(t_flex_rec.device)
                flex_tr_lig_s = (-self.q*torch.log1p(-t_flex_lig)+self.flex_lig_s_min).float().to(t_flex_rec.device)
            elif self.args.flex_tr_t_schedule == "linear":
                flex_tr_rec_s = (T*t_flex_rec + self.flex_rec_s_min).float().to(t_flex_rec.device)
                flex_tr_lig_s = (T*t_flex_lig + self.flex_lig_s_min).float().to(t_flex_rec.device)
            elif self.args.flex_tr_t_schedule == "square":
                flex_tr_rec_s = (T*t_flex_rec**2 + self.flex_rec_s_min).float().to(t_flex_rec.device)
                flex_tr_lig_s = (T*t_flex_lig**2 + self.flex_lig_s_min).float().to(t_flex_rec.device)
            else:
                raise ValueError("flex_tr_t_schedule should be either 'log' or 'linear'or 'square'")

            if self.args.natural_exp:
                flex_rec_s = - torch.expm1(-self.c * t_flex_rec).float().to(t_flex_rec.device)
                flex_lig_s = - torch.expm1(-self.c * t_flex_lig).float().to(t_flex_rec.device)
            else:
                flex_rec_s =torch.clamp(self.flex_rec_s_min + (self.flex_rec_s_max-self.flex_rec_s_min) * (t_flex_rec*self.A_factor) ** self.B_exp, max=torch.tensor(1.).float().to(t_flex_rec.device))
                flex_lig_s =torch.clamp(self.flex_lig_s_min + (self.flex_lig_s_max-self.flex_lig_s_min) * (t_flex_lig*self.A_factor) ** self.B_exp, max=torch.tensor(1.).float().to(t_flex_rec.device))
            
        else:
            if self.args.flex_tr_t_schedule == "log":
                if t_flex_rec < 0.999:
                    flex_tr_rec_s = min(-self.q*math.log1p(-t_flex_rec)+self.flex_rec_s_min, 6.9) # clamp to avoid numerical issues, when t_flex_rec is close to 0.999, the value is close to 6.9
                    flex_tr_lig_s = min(-self.q*math.log1p(-t_flex_lig)+self.flex_lig_s_min, 6.9) 
                else:
                    flex_tr_rec_s = 6.9
                    flex_tr_lig_s = 6.9
            elif self.args.flex_tr_t_schedule == "linear":
                flex_tr_rec_s = T*t_flex_rec+self.flex_rec_s_min # clamp to avoid numerical issues
                flex_tr_lig_s = T*t_flex_lig+self.flex_lig_s_min
            elif self.args.flex_tr_t_schedule == "square":
                flex_tr_rec_s = T*t_flex_rec**2+self.flex_rec_s_min
                flex_tr_lig_s = T*t_flex_lig**2+self.flex_lig_s_min
            else:
                raise ValueError("flex_tr_t_schedule should be either 'log' or 'linear' or 'square'")

            if self.args.natural_exp:
                flex_rec_s = - np.expm1(-self.c * t_flex_rec)
                flex_lig_s = - np.expm1(-self.c * t_flex_lig)
            else:
                flex_rec_s = min(self.flex_rec_s_min + (self.flex_rec_s_max-self.flex_rec_s_min) * (t_flex_rec*self.A_factor) ** self.B_exp, 1.)
                flex_lig_s = min(self.flex_lig_s_min + (self.flex_lig_s_max-self.flex_lig_s_min) * (t_flex_lig*self.A_factor) ** self.B_exp, 1.)


            
        return flex_tr_rec_s, flex_tr_lig_s, flex_rec_s, flex_lig_s, rot_s, tr_s
    


def set_time_flexSDE_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr,
        batch_size: int, device=None):
    """
        Save sampled time to current batch
    """
    lig_size = complex_graphs["ligand"].num_nodes
    complex_graphs["ligand"].node_t = {
        "flex": t_flex_lig * torch.ones(lig_size).to(device),
        "tr": t_tr * torch.ones(lig_size).to(device),
        "rot": t_rot * torch.ones(lig_size).to(device),
    }
    rec_size = complex_graphs["receptor"].num_nodes
    complex_graphs["receptor"].node_t = {
        "flex": t_flex_rec * torch.ones(rec_size).to(device),
        "tr": t_tr * torch.ones(rec_size).to(device),
        "rot": t_rot * torch.ones(rec_size).to(device),
    }
    complex_graphs.complex_t = {
        "flex": t_flex_rec * torch.ones(batch_size).to(device),
        "tr": t_tr * torch.ones(batch_size).to(device),
        "rot": t_rot * torch.ones(batch_size).to(device),
    }


def rotation_matrices_to_vectors(rot_mats):
    """
    Converts a batch of rotation matrices to rotation vectors (axis-angle representation).
    Args:
        rot_mats (torch.Tensor): A batch of rotation matrices of shape (n, 3, 3).
    Returns:
        torch.Tensor: A batch of rotation vectors of shape (n, 3).
    """
    # Ensure input shape is (n, 3, 3)
    assert rot_mats.ndim == 3 and rot_mats.shape[1:] == (3, 3), "Input must have shape (n, 3, 3)"
    
    # Compute the trace of each rotation matrix
    traces = torch.einsum('bii->b', rot_mats)  # Trace of each matrix, shape: (n,)
    
    # Compute the rotation angle using the trace
    cos_theta = (traces - 1) / 2  # cos(theta), shape: (n,)
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # Clamp to avoid numerical issues
    
    # Handle edge case where theta is very small (rotation close to 0)
    near_zero = theta.abs() < 1e-6
    theta[near_zero] = 1  # Temporarily set theta to avoid division by zero
    
    # Compute the skew-symmetric part of the rotation matrix (approximation of axis)
    skew_sym_part = (rot_mats - rot_mats.transpose(1, 2)) / 2  # (n, 3, 3)
    rot_axes = torch.stack([skew_sym_part[:, 2, 1], skew_sym_part[:, 0, 2], skew_sym_part[:, 1, 0]], dim=1)  # (n, 3)
    
    # Normalize rotation axes
    rot_axes = rot_axes / (rot_axes.norm(dim=1, keepdim=True) + 1e-8)  # Avoid division by zero
    
    # Compute the rotation vector (axis scaled by angle)
    rot_vectors = theta.unsqueeze(1) * rot_axes  # (n, 3)
    
    # Handle near-zero rotation
    rot_vectors[near_zero] = 0  # Set to zero for near-zero rotations
    
    return rot_vectors

class SDE:
    def __init__(self, data, args):
        self.args = args
        self.coord_u_receptor = data["receptor"].pos_u[data["receptor"].mask]
        self.coord_u_ligand = data["ligand"].pos_u[data["ligand"].mask]
        self.coord_b_receptor = data["receptor"].pos_b_ori
        self.coord_b_ligand = data["ligand"].pos_b_ori

        self.H_rec = data.H_rec
        self.H_lig = data.H_lig
        del data.H_rec
        del data.H_lig
        self.lambda_rec, self.P_rec = self.get_nonZeroEigen(data["receptor"].Lambda, data["receptor"].P) # non-zero eigenvalues and eigenvectors
        self.lambda_lig, self.P_lig = self.get_nonZeroEigen(data["ligand"].Lambda, data["ligand"].P)
        data["receptor"].smallest_lambda = self.lambda_rec.min()
        data["ligand"].smallest_lambda = self.lambda_lig.min()

        del data["receptor"].Lambda
        del data["receptor"].P
        del data["ligand"].Lambda
        del data["ligand"].P
    
    def get_hessian(self, pos_u):
        """
            Get Hessian matrix for a given set of positions
        """
        # Build the Hessian matrix
        anm = prody.ANM('Protein ANM Analysis')
        anm.buildHessian(pos_u.numpy())
        hessian = anm.getHessian()
        hessian = torch.from_numpy(hessian).float()

        return hessian
    
    def get_nonZeroEigen(self, eigenvalues, eigenvectors, ESP=1e-6, C=1.0, factor=0.1):
        """
            Get eigenvalues and eigenvectors for a given Hessian matrix
        """
        # # Compute the eigenvalues and eigenvectors
        # eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        
        # # Sort non-zero eigenvalues for stability
        # sorted_indices = torch.argsort(eigenvalues)
        # eigenvalues = eigenvalues[sorted_indices]
        # eigenvectors = eigenvectors[:, sorted_indices]

        

        non_zero_eigenvalues = eigenvalues[6:]
        non_zero_eigenvectors = eigenvectors[:, 6:]

        if self.args.eigenscale == "log":
            # scale the eigenvalues by log(lambda_i/lambda_smallest) + C 
            non_zero_eigenvalues = torch.abs(non_zero_eigenvalues)+ESP
            non_zero_eigenvalues = torch.log(non_zero_eigenvalues/non_zero_eigenvalues[0]) + C
        elif self.args.eigenscale == "affine":
            non_zero_eigenvalues = torch.abs(non_zero_eigenvalues)+ESP
            non_zero_eigenvalues = ((non_zero_eigenvalues-non_zero_eigenvalues[0])/(non_zero_eigenvalues[-1] - non_zero_eigenvalues[0]) + C) * factor # value range: [0.1,0.2]

        return non_zero_eigenvalues, non_zero_eigenvectors
    
    def get_train_updates(self, flex_tr_rec_s, flex_tr_lig_s, cutoff=5.0):
        """
            Get train updates (coord_t-coord_0) for a given time t
        """

        # # flex_tr_rec_s, flex_tr_lig_s should be the same
        # s = flex_tr_rec_s

        # # [print_]
        # # print("flex_tr_rec_s", flex_tr_rec_s, "flex_tr_lig_s", flex_tr_lig_s)

        # rec_n = self.coord_u_receptor.shape[0]
        # lig_n = self.coord_u_ligand.shape[0]

        # rec_actInd = self.lambda_rec * s < cutoff
        # lig_actInd = self.lambda_lig * s < cutoff

        # if rec_actInd.sum() == 0:
        #     rec_actInd[0] = True
        #     print("rec_actInd.sum() == 0")

        # if lig_actInd.sum() == 0:
        #     lig_actInd[0] = True
        #     print("lig_actInd.sum() == 0")
        
        # lambda_rec = self.lambda_rec
        # P_rec = self.P_rec
        # lambda_lig = self.lambda_lig
        # P_lig = self.P_lig

        # delta_ca_rec_0 = (self.coord_b_receptor-self.coord_u_receptor).reshape(-1)
        # delta_ca_lig_0 = (self.coord_b_ligand-self.coord_u_ligand).reshape(-1)

        # coeff_rec_0 = P_rec.T @ delta_ca_rec_0
        # coeff_lig_0 = P_lig.T @ delta_ca_lig_0

        # coeff_rec_mean = torch.exp(-0.5*lambda_rec*s) * coeff_rec_0
        # coeff_rec_mean[~rec_actInd] = 0
        # coeff_rec_var = 1/lambda_rec * (-torch.expm1(-lambda_rec*s))
        # coeff_rec_var[~rec_actInd] = 1/lambda_rec[~rec_actInd]

        # coeff_lig_mean = torch.exp(-0.5*lambda_lig*s) * coeff_lig_0
        # coeff_lig_mean[~lig_actInd] = 0
        # coeff_lig_var = 1/lambda_lig * (-torch.expm1(-lambda_lig*s))
        # coeff_lig_var[~lig_actInd] = 1/lambda_lig[~lig_actInd]

        # z_rec = torch.normal(mean=0, std=1, size=(coeff_rec_mean.shape[0],))
        # z_lig = torch.normal(mean=0, std=1, size=(coeff_lig_mean.shape[0],))

        # delta_ca_rec = P_rec @ (coeff_rec_mean + torch.sqrt(coeff_rec_var) * z_rec)
        # delta_ca_lig = P_lig @ (coeff_lig_mean + torch.sqrt(coeff_lig_var) * z_lig)

        # # updates (coord_t-coord_0)
        # flex_rec_update = delta_ca_rec.reshape(-1, 3) + self.coord_u_receptor - self.coord_b_receptor
        # flex_lig_update = delta_ca_lig.reshape(-1, 3) + self.coord_u_ligand - self.coord_b_ligand

        # # scores 
        # flex_rec_score = (- P_rec @ (z_rec/torch.sqrt(coeff_rec_var))).reshape(-1, 3)
        # flex_lig_score = (- P_lig @ (z_lig/torch.sqrt(coeff_lig_var))).reshape(-1, 3)

        # # score_norm
        # flex_rec_score_norm = ((1/coeff_rec_var).sum()/(torch.ones(rec_n)*rec_n))**0.5
        # flex_lig_score_norm = ((1/coeff_lig_var).sum()/(torch.ones(lig_n)*lig_n))**0.5

        # # print(lambda_lig)
        # # print(flex_rec_score, flex_rec_score_norm, flex_rec_update)#, rec_actInd.sum(), lambda_rec, s, flex_rec_score_norm.shape)
        # # print("flex_rec_score_norm: ", flex_rec_score_norm)
        # # print("flex_lig_score_norm: ", flex_lig_score_norm)

        ######## new version ########

        # flex_tr_rec_s, flex_tr_lig_s should be the same
        s = flex_tr_rec_s

        # [print_]
        # print("flex_tr_rec_s", flex_tr_rec_s, "flex_tr_lig_s", flex_tr_lig_s)

        rec_n = self.coord_u_receptor.shape[0]
        lig_n = self.coord_u_ligand.shape[0]
        
        lambda_rec = self.lambda_rec
        P_rec = self.P_rec
        lambda_lig = self.lambda_lig
        P_lig = self.P_lig

        # delta x0
        delta_ca_rec_0 = (self.coord_b_receptor-self.coord_u_receptor).reshape(-1)
        delta_ca_lig_0 = (self.coord_b_ligand-self.coord_u_ligand).reshape(-1)

        # y0
        coeff_rec_0 = P_rec.T @ delta_ca_rec_0
        coeff_lig_0 = P_lig.T @ delta_ca_lig_0

        # yt|y0 mean, var
        

        if self.args.balanced_diffusion:
            coeff_rec_mean = torch.exp(-0.5*lambda_rec*s) * coeff_rec_0
            coeff_rec_var = (-torch.expm1(-lambda_rec*s))
            
            coeff_lig_mean = torch.exp(-0.5*lambda_lig*s) * coeff_lig_0
            coeff_lig_var = (-torch.expm1(-lambda_lig*s))
        else:
            coeff_rec_mean = torch.exp(-0.5*lambda_rec*s) * coeff_rec_0
            coeff_rec_var = 1/lambda_rec * (-torch.expm1(-lambda_rec*s))
            
            coeff_lig_mean = torch.exp(-0.5*lambda_lig*s) * coeff_lig_0
            coeff_lig_var = 1/lambda_lig * (-torch.expm1(-lambda_lig*s))

        # z
        z_rec = torch.normal(mean=0, std=1, size=(coeff_rec_mean.shape[0],))
        z_lig = torch.normal(mean=0, std=1, size=(coeff_lig_mean.shape[0],))

        # yt
        y_t_rec = coeff_rec_mean + torch.sqrt(coeff_rec_var) * z_rec
        y_t_lig = coeff_lig_mean + torch.sqrt(coeff_lig_var) * z_lig

        # delta xt
        delta_ca_rec = P_rec @ (y_t_rec)
        delta_ca_lig = P_lig @ (y_t_lig)

        # updates (coord_t-coord_0)
        flex_rec_update = delta_ca_rec.reshape(-1, 3) + self.coord_u_receptor - self.coord_b_receptor
        flex_lig_update = delta_ca_lig.reshape(-1, 3) + self.coord_u_ligand - self.coord_b_ligand

        # H delta x_t
        H_rec_delta_x_t = (P_rec @ (lambda_rec*y_t_rec)).reshape(-1, 3)
        H_lig_delta_x_t = (P_lig @ (lambda_lig*y_t_lig)).reshape(-1, 3)

        if not self.args.neg_drift:
            # scores 
            flex_rec_score = (- P_rec @ (z_rec/torch.sqrt(coeff_rec_var))).reshape(-1, 3)
            flex_lig_score = (- P_lig @ (z_lig/torch.sqrt(coeff_lig_var))).reshape(-1, 3)

            # score_norm
            flex_rec_score_norm = ((1/coeff_rec_var).sum()/(torch.ones(rec_n)*rec_n))**0.5
            flex_lig_score_norm = ((1/coeff_lig_var).sum()/(torch.ones(lig_n)*lig_n))**0.5

        else:
            # scores 
            flex_rec_score = (- P_rec @ (z_rec/torch.sqrt(coeff_rec_var))).reshape(-1, 3) + H_rec_delta_x_t
            flex_lig_score = (- P_lig @ (z_lig/torch.sqrt(coeff_lig_var))).reshape(-1, 3) + H_lig_delta_x_t

            # print('flex_rec_score norm max mean min:', flex_rec_score.norm(dim=-1).max(), flex_rec_score.norm(dim=-1).mean(), flex_rec_score.norm(dim=-1).min())
            # print('flex_lig_score norm max mean min:', flex_lig_score.norm(dim=-1).max(), flex_lig_score.norm(dim=-1).mean(), flex_lig_score.norm(dim=-1).min())

            # score_norm
            flex_rec_score_norm = H_rec_delta_x_t.norm(dim=-1) # flex_rec_score.norm(dim=-1).mean()/torch.ones(rec_n) # H_rec_delta_x_t.norm(dim=-1) + (((1/coeff_rec_var)).sum()/(torch.ones(rec_n)*rec_n))**0.5 #((torch.exp(-lambda_rec*s)*(1/coeff_rec_var)).sum()/(torch.ones(rec_n)*rec_n))**0.5
            flex_lig_score_norm = H_lig_delta_x_t.norm(dim=-1) # flex_lig_score.norm(dim=-1).mean()/torch.ones(lig_n) # H_lig_delta_x_t.norm(dim=-1) + (((1/coeff_lig_var)).sum()/(torch.ones(lig_n)*lig_n))**0.5 #((torch.exp(-lambda_lig*s)*(1/coeff_lig_var)).sum()/(torch.ones(lig_n)*lig_n))**0.5

            # print('flex_rec_score_norm:', flex_rec_score_norm[0])
            # print('flex_lig_score_norm:', flex_lig_score_norm[0])

        

        # print(lambda_lig)
        # print(flex_rec_score, flex_rec_score_norm, flex_rec_update)#, rec_actInd.sum(), lambda_rec, s, flex_rec_score_norm.shape)
        # print("flex_rec_score_norm: ", flex_rec_score_norm)
        # print("flex_lig_score_norm: ", flex_lig_score_norm)
        


        

        

        return flex_rec_update, flex_lig_update, flex_rec_score, flex_lig_score, flex_rec_score_norm, flex_lig_score_norm
    

        

    

    

