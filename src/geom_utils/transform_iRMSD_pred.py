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
class Transform_iRMSD_pred(BaseTransform):
    """
        Apply translation, rotation, nm noise
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, data):
        """
            Modifies data in place
            @param (torch_geometric.data.HeteroData) data
        """
        # set the num_nodes according to pos_b_ori
        data["receptor"].num_nodes = data["receptor"].pos_b_ori.shape[0]
        data["ligand"].num_nodes = data["ligand"].pos_b_ori.shape[0]

        # data.irmsd
        if "irmsd" in data:
            irmsd = data.irmsd.item()
        else:
            irmsd = compute_irmsd(data['ligand'].pos_b_ori, data["receptor"].pos_b_ori, data['ligand'].pos_u[data['ligand'].mask], data['receptor'].pos_u[data['receptor'].mask], data.name)
        
        irmsd = np.clip(irmsd, a_min=0.35, a_max=8.0)
        data.irmsd = torch.tensor(irmsd).float()

        # interface residue masks [n_res,]; data[key].interface_mask
        ligand_receptor_distance = torch.cdist(data['ligand'].pos_b_ori, data['receptor'].pos_b_ori)
        positive_tuple = torch.where(ligand_receptor_distance < 8.)

        lig_interface_mask = torch.zeros(data['ligand'].pos_b_ori.size(0), dtype=torch.bool, device=data['ligand'].pos_b_ori.device)
        rec_interface_mask = torch.zeros(data['receptor'].pos_b_ori.size(0), dtype=torch.bool, device=data['receptor'].pos_b_ori.device)

        lig_interface_mask[positive_tuple[0]] = True
        rec_interface_mask[positive_tuple[1]] = True

        data['ligand'].interface_mask = lig_interface_mask
        data['receptor'].interface_mask = rec_interface_mask

        # pos and x
        data['receptor'].pos = data['receptor'].pos_u[data['receptor'].mask]
        data['ligand'].pos = data['ligand'].pos_u[data['ligand'].mask]

        if 'x_ori' in data["receptor"]:
            data["receptor"].x = data["receptor"].x_ori
            data["ligand"].x = data["ligand"].x_ori


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

        return data

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


    
