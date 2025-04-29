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

from .so3 import sample_vec, score_vec
from .torus import score as score_torus
from .geometry import kabsch_torch, axis_angle_to_matrix

EPS = 1e-6


# check if anywhere the functions are used !!!!

# ------ PyG UTILS -------
class NoiseTransform_flexFM_Rt(BaseTransform):
    """
        Apply translation, rotation, nm noise
    """
    def __init__(self, args):
        # save min/max sigma scales
        self.noise_schedule = NoiseSchedule_flexFM_Rt(args)
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

        # [adaptive_schedule]
        if self.args.adaptive_schedule:
            if "irmsd" in data:
                irmsd = data.irmsd.item()
            else:
                irmsd = compute_irmsd(data['ligand'].pos_b_ori, data["receptor"].pos_b_ori, data['ligand'].pos_u[data['ligand'].mask], data['receptor'].pos_u[data['receptor'].mask], data.name)
            
            alpha = 0.99
            adaptive_c = ((-np.log(1-alpha)*np.log(3000.0))/(np.log(np.clip(irmsd, a_min=0.35, a_max=8.0))+np.log(100.0))) # c=10.4 to c=5.5
            data.adaptive_c = adaptive_c

            flex_rec_s, flex_lig_s, rot_s, tr_s = self.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr, adaptive_c=adaptive_c)

            
        
        else:
            if "irmsd" in data:
                del data.irmsd
            flex_rec_s, flex_lig_s, rot_s, tr_s = self.noise_schedule(t_flex_rec, t_flex_lig, t_rot, t_tr)

        training_flag = False

        if flex_rec_update is None: # in training
            training_flag = True

        # set the num_nodes according to pos_b_ori
        data["receptor"].num_nodes = data["receptor"].pos_b_ori.shape[0]
        data["ligand"].num_nodes = data["ligand"].pos_b_ori.shape[0]

        # [adaptive_schedule]
        if self.args.adaptive_schedule:
            set_time_flexFM_Rt(data, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device=None, t_fm_deform_rec=flex_rec_s, t_fm_deform_lig=flex_lig_s)
        else:
            set_time_flexFM_Rt(data, t_flex_rec, t_flex_lig, t_rot, t_tr, 1, device=None)

        del data.H_rec
        del data.H_lig
        del data["receptor"].Lambda
        del data["receptor"].P
        del data["ligand"].Lambda
        del data["ligand"].P

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
            # u(t) tr_rec*
            flex_rec_update = (data['receptor'].pos_u[data['receptor'].mask]-data['receptor'].pos_b_ori) * rec_sigma  # + torch.normal(mean=0, std=0.3, size=(data['receptor'].pos_b_ori.shape[0], 3))
            # u(t) tr_lig* 
            flex_lig_update = (data['ligand'].pos_u[data['ligand'].mask]-data['ligand'].pos_b_ori) * lig_sigma  # + torch.normal(mean=0, std=0.3, size=(data['ligand'].pos_b_ori.shape[0], 3))
            
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

            if self.args.use_nm_msf_feat:
                # calculate msf
                data["receptor"].cur_nm = data["receptor"].b_nm # Shape: (n_res, 3, n_NM)
                data["receptor"].eig_vals = data["receptor"].eig_vals_b.float().reshape(1, -1) # Shape: (1, n_NM)
                msf = compute_msf(data["receptor"].cur_nm, data["receptor"].eig_vals.reshape(-1,))

                # q2 = torch.quantile(msf, 0.02)
                # q98 = torch.quantile(msf, 0.98)
                # msf_clipped = torch.clamp(msf, q2, q98)
                # data["receptor"].msf = msf_clipped # min_val + (msf_clipped - q5) / (q95 - q5) * (max_val - min_val)

                msf_log = torch.log(msf + 1e-6)  # add a small constant to avoid log(0)
                msf_normalized = (msf_log - msf_log.mean()) / msf_log.std()
                data["receptor"].msf = msf_normalized

                
                
                data["ligand"].cur_nm = data["ligand"].b_nm # Shape: (n_res, 3, n_NM)
                data["ligand"].eig_vals = data["ligand"].eig_vals_b.float().reshape(1, -1) # Shape: (1, n_NM)
                data["ligand"].msf = compute_msf(data["ligand"].cur_nm, data["ligand"].eig_vals.reshape(-1,))

                # q2 = torch.quantile(data["ligand"].msf, 0.02)
                # q98 = torch.quantile(data["ligand"].msf, 0.98)
                # msf_clipped = torch.clamp(data["ligand"].msf, q2, q98)
                # data["ligand"].msf = msf_clipped # min_val + (msf_clipped - q5) / (q95 - q5) * (max_val - min_val)

                msf_log = torch.log(data["ligand"].msf + 1e-6)  # add a small constant to avoid log(0)
                msf_normalized = (msf_log - msf_log.mean()) / msf_log.std()
                data["ligand"].msf = msf_normalized

                # print max and min msf
                # print(f"receptor max msf: {data['receptor'].msf.max()}; min msf: {data['receptor'].msf.min()}")
                # print(f"ligand max msf: {data['ligand'].msf.max()}; min msf: {data['ligand'].msf.min()}")

            


            if 'H_lig' in data:
                del data.H_lig
                del data.H_rec
            if 'Lambda' in data["receptor"]:
                del data["receptor"].Lambda
                del data["receptor"].P
                del data["ligand"].Lambda
                del data["ligand"].P
            if 'ts_rec' in data:
                del data.ts_rec
                del data.ts_lig
                
        if self.args.res_rot_flex:
            # apply updates
            self.apply_updates(data, flex_rec_update=flex_rec_update,
                            flex_lig_update=flex_lig_update,
                            rot_update=rot_update,
                            tr_update=tr_update,
                            flex_rec_rot_update=flex_rec_rot_update,
                            flex_lig_rot_update=flex_lig_rot_update)

            # compute ground truth score given updates, noise level
            self.get_score(data, flex_rec_update, flex_lig_update,
                        rot_update, tr_update, flex_rec_rot_update=flex_rec_rot_update,
                        flex_lig_rot_update=flex_lig_rot_update, orig_ca_lig_cross_distances=orig_ca_lig_cross_distances)

        else:
            # apply updates
            self.apply_updates(data, flex_rec_update=flex_rec_update,
                            flex_lig_update=flex_lig_update,
                            rot_update=rot_update,
                            tr_update=tr_update)

            # compute ground truth score given updates, noise level
            self.get_score(data, flex_rec_update, flex_lig_update,
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
        # kabsch to separate the residue-level translation and the global rigid-body motions
        R, t = kabsch_torch(new_pos_l.T, flex_new_pos_l.T)
        new_pos_l = new_pos_l @ R.T + t.T
        flex_lig_update = new_pos_l - flex_new_pos_l

        com = torch.mean(new_pos_l, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())

        # print("new_pos_l dtype:", new_pos_l.dtype)
        # print("com dtype:", com.dtype)
        # print("rot_mat dtype:", rot_mat.dtype)
        # print("tr_update dtype:", tr_update.dtype)


        new_pos_l = (new_pos_l -com) @ rot_mat.T + com + tr_update

        #
        # rec Ca pos: flex
        new_pos_r = (
            flex_new_pos_r + flex_rec_update
        )
        # kabsch to separate the residue-level translation and the global rigid-body motions
        R, t = kabsch_torch(new_pos_r.T, flex_new_pos_r.T)
        new_pos_r = new_pos_r @ R.T + t.T
        flex_rec_update = new_pos_r - flex_new_pos_r

        
        data["ligand"].pos = new_pos_l
        data["receptor"].pos = new_pos_r

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

    def get_score(self, data, flex_rec_update, flex_lig_update,
                  rot_update, tr_update, flex_rec_rot_update=None,
                  flex_lig_rot_update=None,orig_ca_lig_cross_distances=None):
        """
            Compute ground truth score, given updates and noise.
            Modifies data in place.
        """
        # rec flex: -u(t) tr_rec*
        data.flex_rec_score = -flex_rec_update
        # lig flex: -u(t) tr_lig* @ R_t
        # com = torch.mean(flex_lig_update, dim=0, keepdim=True)
        rot_mat = axis_angle_to_matrix(rot_update.squeeze())
        # flex_lig_update = (flex_lig_update -com) @ rot_mat.T + com
        flex_lig_update = flex_lig_update @ rot_mat.T
        data.flex_lig_score = -flex_lig_update

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


class NoiseSchedule_flexFM_Rt:
    """
        Transforms t into scaled sigmas
        Note: the nm components for t and sigma should be same for r/l
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


    def __call__(self, t_flex_rec, t_flex_lig, t_rot, t_tr, adaptive_c=None):
        """
            Convert from time to (scaled) sigma space
        """
        tr_s = self.tr_s_min ** (1 - t_tr) * self.tr_s_max**t_tr
        rot_s = self.rot_s_min ** (1 - t_rot) * self.rot_s_max**t_rot

        if torch.is_tensor(t_flex_rec):
            if self.args.natural_exp:
                if adaptive_c is None: # [adaptive_schedule]
                    flex_rec_s = 1 - torch.exp(-self.c * t_flex_rec).float().to(t_flex_rec.device)
                    flex_lig_s = 1 - torch.exp(-self.c * t_flex_lig).float().to(t_flex_rec.device)
                else:
                    flex_rec_s = 1 - torch.exp(-adaptive_c * t_flex_rec).float().to(t_flex_rec.device)
                    flex_lig_s = 1 - torch.exp(-adaptive_c * t_flex_lig).float().to(t_flex_rec.device)
            else:
                flex_rec_s =torch.clamp(self.flex_rec_s_min + (self.flex_rec_s_max-self.flex_rec_s_min) * (t_flex_rec*self.A_factor) ** self.B_exp, max=torch.tensor(1.).float().to(t_flex_rec.device))
                flex_lig_s =torch.clamp(self.flex_lig_s_min + (self.flex_lig_s_max-self.flex_lig_s_min) * (t_flex_lig*self.A_factor) ** self.B_exp, max=torch.tensor(1.).float().to(t_flex_rec.device))
        else:
            if self.args.natural_exp:
                if adaptive_c is None: # [adaptive_schedule]
                    flex_rec_s = 1 - np.exp(-self.c * t_flex_rec)
                    flex_lig_s = 1 - np.exp(-self.c * t_flex_lig)
                else:
                    # check adaptive_c and make sure it be a numpy array
                    if isinstance(adaptive_c, torch.Tensor):
                        adaptive_c = adaptive_c.cpu().numpy()

                    flex_rec_s = 1 - np.exp(-adaptive_c * t_flex_rec)
                    flex_lig_s = 1 - np.exp(-adaptive_c * t_flex_lig)
            else:
                flex_rec_s = min(self.flex_rec_s_min + (self.flex_rec_s_max-self.flex_rec_s_min) * (t_flex_rec*self.A_factor) ** self.B_exp, 1.)
                flex_lig_s = min(self.flex_lig_s_min + (self.flex_lig_s_max-self.flex_lig_s_min) * (t_flex_lig*self.A_factor) ** self.B_exp, 1.)

        return flex_rec_s, flex_lig_s, rot_s, tr_s
    

# [adaptive_schedule]
def set_time_flexFM_Rt(complex_graphs, t_flex_rec, t_flex_lig, t_rot, t_tr,
        batch_size: int, device=None, t_fm_deform_rec=None, t_fm_deform_lig=None):
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

    if t_fm_deform_rec is not None:
        # check t_fm_deform_rec and t_fm_deform_lig and make sure they are torch tensors
        if not torch.is_tensor(t_fm_deform_rec):
            t_fm_deform_rec = torch.tensor(t_fm_deform_rec).to(device)
        if not torch.is_tensor(t_fm_deform_lig):
            t_fm_deform_lig = torch.tensor(t_fm_deform_lig).to(device)

        if t_fm_deform_rec.ndim != 0:
            batch_rec = complex_graphs["receptor"].batch
            complex_graphs["receptor"].t_fm_deform = t_fm_deform_rec[batch_rec].to(device)
            batch_lig = complex_graphs["ligand"].batch
            complex_graphs["ligand"].t_fm_deform = t_fm_deform_lig[batch_lig].to(device)
        else:
            complex_graphs["ligand"].t_fm_deform = t_fm_deform_rec * torch.ones(lig_size).to(device)
            complex_graphs["receptor"].t_fm_deform = t_fm_deform_lig * torch.ones(rec_size).to(device)
            
        complex_graphs.t_fm_deform = t_fm_deform_rec * torch.ones(batch_size).to(device)
        



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


def compute_msf(normal_mode_matrix, eigenvalues):
    ''' Compute the mean squared fluctuations (MSF) of a protein structure using normal modes.
    Args:
        normal_mode_matrix (torch.Tensor): The normal mode matrix of shape (n_res, 3, n_NM).
        eigenvalues (torch.Tensor): The eigenvalues of the normal modes of shape (n_NM,).
    Returns:
        torch.Tensor: The MSF of each residue in the protein structure of shape (n_res,).
    '''
    # Ensure eigenvalues are reshaped for broadcasting
    inv_eigenvalues = 1.0 / eigenvalues  # Shape: (n_NM,)

    # Compute the squared displacements: sum over x, y, z for each residue and mode
    squared_displacements = torch.einsum("rij,rij->rj", normal_mode_matrix, normal_mode_matrix)
    # Resulting shape: (n_res, n_NM)

    # Weight by the inverse eigenvalues and sum over modes (n_NM)
    msf = torch.einsum("rj,j->r", squared_displacements, inv_eigenvalues)
    # Resulting shape: (n_res,)
    return msf




def compute_irmsd(ligand_coors_true_ori, receptor_coors_true_ori, ligand_coors_pred, receptor_coors_pred, name):
    ''' Compute the interface root-mean-square deviation (iRMSD) between two protein structures.
    Args:
        ligand_coors_true_ori (torch.Tensor): The original coordinates of the ligand structure of shape (n_res_lig, 3).
        receptor_coors_true_ori (torch.Tensor): The original coordinates of the receptor structure of shape (n_res_rec, 3).
        ligand_coors_pred (torch.Tensor): The predicted coordinates of the ligand structure of shape (n_res_lig, 3).
        receptor_coors_pred (torch.Tensor): The predicted coordinates of the receptor structure of shape (n_res_rec, 3).
    Returns:
        torch.Tensor: The iRMSD between the two protein structures.
    '''
    # Compute the pairwise distances between ligand and receptor residues
    ligand_receptor_distance = torch.cdist(ligand_coors_true_ori, receptor_coors_true_ori)
    positive_tuple = torch.where(ligand_receptor_distance < 8.)

    if len(positive_tuple[0]) == 0:
        positive_tuple = torch.where(ligand_receptor_distance < 15.)
        print(f"There is no interface residue in the complex {name} with cutoff 8. Please check the input data.")

        

    # if len(positive_tuple[0]) == 0:
    #     print(f"There is no interface residue in the complex. Please check the input data.")
    

    # Remove duplicates while preserving order
    active_ligand, unique_ligand_indices = torch.unique(positive_tuple[0], return_inverse=True)
    active_receptor, unique_receptor_indices = torch.unique(positive_tuple[1], return_inverse=True)

    
    ligand_coors_pred = ligand_coors_pred[active_ligand, :]
    ligand_coors_true = ligand_coors_true_ori[active_ligand, :]

    receptor_coors_pred = receptor_coors_pred[active_receptor, :]
    receptor_coors_true = receptor_coors_true_ori[active_receptor, :]
    

    complex_coors_pred = torch.cat((ligand_coors_pred, receptor_coors_pred), dim=0)
    complex_coors_true = torch.cat((ligand_coors_true, receptor_coors_true), dim=0)

    R,t = kabsch_torch(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    interface_rmsd = torch.sqrt(torch.mean((complex_coors_pred_aligned - complex_coors_true) ** 2))
    return interface_rmsd


    
