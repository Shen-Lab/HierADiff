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

# ------ PyG UTILS -------
class NoiseTransform_nm(BaseTransform):
    """
        Apply translation, rotation, nm noise
    """
    def __init__(self, args):
        # save min/max sigma scales
        self.noise_schedule = NoiseSchedule_nm(args)
        self.args = args

    def __call__(self, data):
        """
            Modifies data in place
            @param (torch_geometric.data.HeteroData) data
        """
        t = np.random.uniform()  # sample time
        t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig = t, t, t, t  # same time scale for each
        data = self.apply_noise(data, t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig)
        return data

    def apply_noise(self, data, t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig,
                    tr_rec_update=None,
                    rot_rec_update=None,
                    tr_lig_update=None,
                    rot_lig_update=None):
        """
            Apply noise to existing HeteroData object
            @param (torch_geometric.data.HeteroData) data
        """

        tr_rec_s, rot_rec_s, tr_lig_s, rot_lig_s = self.noise_schedule(t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig)

        training_flag = False

        if tr_rec_update is None: # in training
            training_flag = True

        # set the num_nodes according to pos_b_ori
        data["receptor"].num_nodes = data["receptor"].pos_b_ori.shape[0]
        data["ligand"].num_nodes = data["ligand"].pos_b_ori.shape[0]
            
        set_time_nm(data, t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig, 1, device=None)

        # sample updates if not provided (in training)
        if training_flag:

            # if tr_update is None: # this must be true (in training)
            #     tr_update = torch.normal(mean=0, std=tr_s, size=(1, 3))
            # if rot_update is None: # this must be true (in training)
            #     rot_update = sample_vec(eps=rot_s)
            #     rot_update = torch.from_numpy(rot_update).float()

            # res_decay_weight: residues to another protein center distance decay weight ?
            lig_center = data['ligand'].pos_b_ori.mean(dim=0,keepdim=True)
            rec_center = data['receptor'].pos_b_ori.mean(dim=0,keepdim=True)
            lig_distance = (data['ligand'].pos_b_ori-rec_center).norm(dim=-1,keepdim=True)
            rec_distance = (data['receptor'].pos_b_ori-lig_center).norm(dim=-1,keepdim=True)
            data['receptor'].res_decay_weight = torch.exp(-torch.nn.ReLU()((rec_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))
            data['ligand'].res_decay_weight = torch.exp(-torch.nn.ReLU()((lig_distance-15.)/20.)) # torch.exp(-torch.nn.ReLU()((res_distance-6.)/10.))




            orig_ca_lig_cross_distances = (data['ligand'].pos_b_ori[None,...] - data['receptor'].pos_b_ori[:,None,...]).norm(dim=-1)

            # random translation and rotation for ub ligand
            lig_rand_rot = sample_vec(eps=self.args.rot_s_max)
            lig_rand_rot = torch.from_numpy(lig_rand_rot).float()

            lig_rand_tr = torch.normal(mean=0, std=self.args.tr_s_max, size=(1, 3))

            flex_new_pos_l = data["ligand"].pos_u[data["ligand"].mask]
            com = torch.mean(flex_new_pos_l, dim=0, keepdim=True)
            rot_mat = axis_angle_to_matrix(lig_rand_rot.squeeze())

            new_pos_l = (flex_new_pos_l -com) @ rot_mat.T + lig_rand_tr + com

            # pos_bb_atoms_u_lig = data['ligand'].bb_atoms_u[data["ligand"].mask]

            # new_pos_bb_atoms_u_lig = (
            #     (pos_bb_atoms_u_lig.reshape((-1, 3)) - com) @ rot_mat.T + lig_rand_tr + com
            # )
            # new_pos_bb_atoms_u_lig = new_pos_bb_atoms_u_lig.reshape((-1, 3, 3))

            # # # bb_atoms
            # # data['ligand'].bb_atoms_u_t = item[f"{key}_u_bb_coords"].float()
            

            # v_t
            rec_sigma = torch.clamp(tr_rec_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise
            lig_sigma = torch.clamp(tr_lig_s + torch.normal(mean=0., std=0.2, size=(1,)).float(),min=0., max=1.)[0] # interpolated + some rand_noise 

            # current tr and res_loss_weight
            tr_update_norm = (lig_rand_tr * lig_sigma).norm(dim=-1).item()
            data['receptor'].res_loss_weight = torch.ones(data['receptor'].pos_b_ori.shape[0],1).float() * (tr_update_norm<15.) # torch.ones(data['receptor'].pos.shape[0],1).float() * (tr_update_norm<6.)
            data['ligand'].res_loss_weight = torch.ones(data['ligand'].pos_b_ori.shape[0],1).float() # * (tr_update_norm<15.) # torch.ones(data['receptor'].pos.shape[0],1).float() * (tr_update_norm<6.)

            # print((data['receptor'].b_to_u_nmCoeffs * nm_sigma).shape, (data['receptor'].b_to_u_nmCoeffs).shape)
            tr_rec_update = (data['receptor'].pos_u[data['receptor'].mask]-data['receptor'].pos_b_ori) * rec_sigma  # + torch.normal(mean=0, std=0.3, size=(data['receptor'].pos_b_ori.shape[0], 3))
            tr_lig_update = (new_pos_l-data['ligand'].pos_b_ori) * lig_sigma  # + torch.normal(mean=0, std=0.3, size=(data['ligand'].pos_b_ori.shape[0], 3))

    
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
                

        # apply updates
        self.apply_updates(data, tr_rec_update=tr_rec_update,rot_rec_update=rot_rec_update,tr_lig_update=tr_lig_update,rot_lig_update=rot_lig_update)

        # compute ground truth score given updates, noise level
        self.get_score(data, tr_rec_update, rot_rec_update,
                       tr_lig_update, rot_lig_update, orig_ca_lig_cross_distances=orig_ca_lig_cross_distances)

        return data

    def apply_updates(self, data,
                      tr_rec_update=None,rot_rec_update=None,tr_lig_update=None,rot_lig_update=None):
        """
            Apply translation / rotation to
            data["ligand"].pos (shape: (n_res, 3)) and data["receptor"].pos (shape: (n_res, 3))
            @param (torch_geometric.data.HeteroData) data
        """

        # print("coord_r_updates", coord_r_updates)
        # print("coord_l_updates", coord_l_updates)

        # defaulted
        flex_new_pos_l = data["ligand"].pos
        flex_new_pos_r = data["receptor"].pos


        new_pos_l = (
            flex_new_pos_l + tr_lig_update
        )

        new_pos_r = (
            flex_new_pos_r + tr_rec_update
        )

        

        data["ligand"].pos = new_pos_l
        data["receptor"].pos = new_pos_r

        

        return data

    def get_score(self, data, tr_rec_update, rot_rec_update, tr_lig_update, rot_lig_update, orig_ca_lig_cross_distances=None):
        """
            Compute ground truth score, given updates and noise.
            Modifies data in place.
        """
        
        data.tr_rec_score = -tr_rec_update
        data.rot_rec_score = None
        data.tr_lig_score = -tr_lig_update
        data.rot_lig_score = None

        # clddt
        if orig_ca_lig_cross_distances is not None:
            ca_lig_cross_distances = (data['ligand'].pos[None,...] - data['receptor'].pos[:,None,...]).norm(dim=-1)
            ca_lig_cross_distances_diff = (orig_ca_lig_cross_distances - ca_lig_cross_distances).abs()

            cutoff_mask = (orig_ca_lig_cross_distances < 15.0).float()
            score = 0.25 * ((ca_lig_cross_distances_diff<0.5).float()
                            + (ca_lig_cross_distances_diff<1.0).float()
                            + (ca_lig_cross_distances_diff<2.0).float()
                            + (ca_lig_cross_distances_diff<4.0).float())
            ca_lig_cross_lddt =  (score * cutoff_mask).sum() / cutoff_mask.sum()
            data.lddt = ca_lig_cross_lddt[None,None]

        return data


class NoiseSchedule_nm:
    """
        Transforms t into scaled sigmas
        Note: the nm components for t and sigma should be same for r/l
    """
    def __init__(self, args):
        self.tr_rec_s_min = args.tr_rec_s_min
        self.tr_rec_s_max = args.tr_rec_s_max
        self.rot_rec_s_max = args.rot_rec_s_max
        self.rot_rec_s_min = args.rot_rec_s_min
        self.tr_lig_s_min = args.tr_lig_s_min
        self.tr_lig_s_max = args.tr_lig_s_max
        self.rot_lig_s_max = args.rot_lig_s_max
        self.rot_lig_s_min = args.rot_lig_s_min

    def __call__(self, t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig):
        """
            Convert from time to (scaled) sigma space
        """
        # tr_s = self.tr_s_min ** (1 - t_tr) * self.tr_s_max**t_tr
        # rot_s = self.rot_s_min ** (1 - t_rot) * self.rot_s_max**t_rot

        if torch.is_tensor(t_tr_rec):
            tr_rec_s =torch.clamp(self.tr_rec_s_min + (self.tr_rec_s_max-self.tr_rec_s_min) * (t_tr_rec*5.) ** 0.3, max=torch.tensor(1.).float().to(t_tr_rec.device))
            rot_rec_s = torch.clamp(self.rot_rec_s_min + (self.rot_rec_s_max-self.rot_rec_s_min) * (t_rot_rec*5.) ** 0.3, max=torch.tensor(1.).float().to(t_tr_rec.device))
            tr_lig_s = torch.clamp(self.tr_lig_s_min + (self.tr_lig_s_max-self.tr_lig_s_min) * (t_tr_lig*5.) ** 0.3, max=torch.tensor(1.).float().to(t_tr_rec.device))
            rot_lig_s = torch.clamp(self.rot_lig_s_min + (self.rot_lig_s_max-self.rot_lig_s_min) * (t_rot_lig*5.) ** 0.3, max=torch.tensor(1.).float().to(t_tr_rec.device))
        else:
            tr_rec_s = min(self.tr_rec_s_min + (self.tr_rec_s_max-self.tr_rec_s_min) * (t_tr_rec*5.) ** 0.3, 1.)
            rot_rec_s = min(self.rot_rec_s_min + (self.rot_rec_s_max-self.rot_rec_s_min) * (t_rot_rec*5.) ** 0.3, 1.)
            tr_lig_s = min(self.tr_lig_s_min + (self.tr_lig_s_max-self.tr_lig_s_min) * (t_tr_lig*5.) ** 0.3, 1.)
            rot_lig_s = min(self.rot_lig_s_min + (self.rot_lig_s_max-self.rot_lig_s_min) * (t_rot_lig*5.) ** 0.3, 1.)


        return tr_rec_s, rot_rec_s, tr_lig_s, rot_lig_s
    


def set_time_nm(complex_graphs, t_tr_rec, t_rot_rec, t_tr_lig, t_rot_lig,
        batch_size: int, device=None):
    """
        Save sampled time to current batch
    """
    lig_size = complex_graphs["ligand"].num_nodes
    complex_graphs["ligand"].node_t = {
        "tr": t_tr_lig * torch.ones(lig_size).to(device),
        "rot": t_rot_lig * torch.ones(lig_size).to(device),
    }
    rec_size = complex_graphs["receptor"].num_nodes
    complex_graphs["receptor"].node_t = {
        "tr": t_tr_rec * torch.ones(rec_size).to(device),
        "rot": t_rot_rec * torch.ones(rec_size).to(device),
    }
    complex_graphs.complex_t = {
        "tr": t_tr_lig * torch.ones(batch_size).to(device),
        "rot": t_rot_lig * torch.ones(batch_size).to(device),
    }

