import math
import pandas as pd
import os

from biopandas.pdb import PandasPdb
import numpy as np

import torch

from matplotlib import pyplot as plt
import scipy.spatial as spa
from tqdm import tqdm

# [nm]
# Input: expects 3xN matrix of points
# Returns such R, t so that rmsd(R @ A + t, B) is min
# Uses Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm)
# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

def compute_rmsd_align(pred, true):
    R,t = rigid_transform_Kabsch_3D(pred.T, true.T)
    pred_aligned = (R @ pred.T + t).T
    return compute_rmsd(pred_aligned, true)

# [nm]
def compute_rmsd(pred, true):
    return np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1)))

# [nm]
def get_rmsd_summary(rmsds, percentage=False):
    rmsds_np = np.array(rmsds)
    if percentage:
        return {
            'mean': np.mean(rmsds_np) * 100,
            'median': np.median(rmsds_np) * 100,
            'std': np.std(rmsds_np) * 100,
            'perc_above1': 100 * (rmsds_np > 0.01).sum() / len(rmsds_np),
            'perc_above2': 100 * (rmsds_np > 0.02).sum() / len(rmsds_np),
            'perc_above5': 100 * (rmsds_np > 0.05).sum() / len(rmsds_np),
            'perc_above10': 100 * (rmsds_np > 0.1).sum() / len(rmsds_np),
            'perc_above30': 100 * (rmsds_np > 0.2).sum() / len(rmsds_np),
            'perc_above50': 100 * (rmsds_np > 0.5).sum() / len(rmsds_np),
            'perc_above80': 100 * (rmsds_np > 0.8).sum() / len(rmsds_np)
        }
    return {
        'mean': np.mean(rmsds_np),
        'median': np.median(rmsds_np),
        'std': np.std(rmsds_np),
        'lt1': 100 * (rmsds_np < 1.0).sum() / len(rmsds_np),
        'lt2': 100 * (rmsds_np < 2.0).sum() / len(rmsds_np),
        'lt5': 100 * (rmsds_np < 5.0).sum() / len(rmsds_np),
        'lt10': 100 * (rmsds_np < 10.0).sum() / len(rmsds_np)
    }

def get_quality_summary(quality_list):
    quality_np = np.array(quality_list)
    return {
        'high': 100 * (quality_np == 'high').sum() / len(quality_np),
        'medium': 100 * (quality_np == 'medium').sum() / len(quality_np),
        'acceptable': 100 * (quality_np == 'acceptable').sum() / len(quality_np),
        'incorrect': 100 * (quality_np == 'incorrect').sum() / len(quality_np)
    }


# [nm]
class RMSDComputer():
    def __init__(self):
        # final results
        self.complex_rmsd_list = []
        self.ligand_rmsd_list = []
        self.interface_rmsd_list = []

        self.ind_rmsd_list = []

        self.pred_to_b_rmsd_list = []
        self.u_to_true_rmsd_list = []

        self.recovery_rmsd_prec_ligand_list = []
        self.recovery_rmsd_prec_receptor_list = []
        # final results
        self.recovery_rmsd_prec_list = []

        self.recovery_interface_rmsd_prec_ligand_list = []
        self.recovery_interface_rmsd_prec_receptor_list = []
        self.recovery_interface_rmsd_prec_list = []

        self.true_lddt_list = []

        self.fnat_list = []

        self.docking_quality_list = []

        self.names = []


    
    def update_all_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, ligand_coors_u=None, receptor_coors_u=None, receptor_coors_true_ori=None, ligand_coors_true_ori=None, name = None):
        if name is not None:
            self.names.append(name)
        
        complex_rmsd = self.update_complex_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true)
        ligand_rmsd = self.update_ligand_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true)
        interface_rmsd = self.update_interface_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, receptor_coors_true_ori, ligand_coors_true_ori, cutoff=10.0)
        ind_rmsd_pair = self.update_ind_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true)

        lddt = self.update_lddt(ligand_coors_pred, receptor_coors_pred, ligand_coors_true_ori, receptor_coors_true_ori)
        fnat = self.update_fnat(ligand_coors_pred, receptor_coors_pred, ligand_coors_true_ori, receptor_coors_true_ori, cutoff=8.0)

        # print("ligand_rmsd, interface_rmsd, fnat: ", ligand_rmsd, interface_rmsd, fnat)
        docking_quality = self.update_docking_quality(ligand_rmsd, interface_rmsd, fnat)


        recovery_rmsd_prec = None
        interface_rmsd_prec = None
        if ligand_coors_u is not None:
            pred_to_b_ligand = compute_rmsd_align(ligand_coors_pred, ligand_coors_true)
            pred_to_b_receptor = compute_rmsd_align(receptor_coors_pred, receptor_coors_true)

            u_to_true_ligand = compute_rmsd_align(ligand_coors_u, ligand_coors_true)
            u_to_true_receptor = compute_rmsd_align(receptor_coors_u, receptor_coors_true)
            
            recovery_rmsd_prec_ligand = 1 - pred_to_b_ligand/u_to_true_ligand
            recovery_rmsd_prec_receptor = 1 - pred_to_b_receptor/u_to_true_receptor

            recovery_rmsd_prec = (recovery_rmsd_prec_ligand + recovery_rmsd_prec_receptor)/2

            self.recovery_rmsd_prec_ligand_list.append(1 - pred_to_b_ligand/u_to_true_ligand)
            self.recovery_rmsd_prec_receptor_list.append(1 - pred_to_b_receptor/u_to_true_receptor)

            self.recovery_rmsd_prec_list.append((recovery_rmsd_prec_ligand + recovery_rmsd_prec_receptor)/2)

            ### old ###
            
            # pred_to_b_rmsd = self.update_complex_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, self_list=self.pred_to_b_rmsd_list)
            # u_to_true_rmsd = self.update_complex_rmsd(ligand_coors_u, ligand_coors_true, receptor_coors_u, receptor_coors_true, self_list=self.u_to_true_rmsd_list)
            
            # recovery_rmsd_prec = 1 - pred_to_b_rmsd/u_to_true_rmsd
            # self.recovery_rmsd_prec_list.append(recovery_rmsd_prec)

            ### old ###

            self.update_interface_rmsd_recovery(ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, receptor_coors_u, ligand_coors_u, cutoff=10.0)


        return complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec
    
    def update_docking_quality(self, ligand_rmsd, interface_rmsd, fnat):
        
        if fnat >= 0.5 and (ligand_rmsd <= 1.0 or interface_rmsd <= 1.0):
            docking_quality = 'high'
        elif fnat >= 0.3 and (ligand_rmsd <= 5.0 or interface_rmsd <= 2.0):
            docking_quality = 'medium'
        elif fnat >= 0.1 and (ligand_rmsd <= 10.0 or interface_rmsd <= 4.0):
            docking_quality = 'acceptable'
        else:
            docking_quality = 'incorrect'

        self.docking_quality_list.append(docking_quality)

        return docking_quality

    def update_fnat(self, ligand_coors_pred, receptor_coors_pred, ligand_coors_true_ori, receptor_coors_true_ori, cutoff=5.0):
        """
        Compute the fraction of native contacts (f_nat) for the ligand–receptor interface,
        assuming the inputs are NumPy arrays.
        fnat=(predicted contacts ∩ true contacts)/(true contacts)
        """
        ligand_receptor_distance = spa.distance.cdist(ligand_coors_true_ori, receptor_coors_true_ori)
        true_positive_tuple = np.where(ligand_receptor_distance < cutoff)

        # print(f"complex {self.names[-1]} has {len(true_positive_tuple[0])} contacts (cutoff: {cutoff})")
        if len(true_positive_tuple[0]) == 0:
            print(f"There is no contacts (cutoff: {cutoff}) in the complex {self.names[-1]}. Please check the input data. use cutoff = 10.0 instead.")
            cutoff = 10.0
            true_positive_tuple = np.where(ligand_receptor_distance < cutoff)

        
        n_true_contacts = len(true_positive_tuple[0])

        # true contacts in the predicted contacts
        ligand_receptor_distance_pred = spa.distance.cdist(ligand_coors_pred, receptor_coors_pred)
        pred_positive_tuple = np.where(ligand_receptor_distance_pred < cutoff)

        # the overlap between true contacts and predicted contacts
        n_overlap = len(set(zip(true_positive_tuple[0], true_positive_tuple[1])) & set(zip(pred_positive_tuple[0], pred_positive_tuple[1])))

        fnat = n_overlap/n_true_contacts

        self.fnat_list.append(fnat)

        return fnat


        

        
        return fnat
        


    def update_lddt(self, ligand_coors_pred, receptor_coors_pred, ligand_coors_true_ori, receptor_coors_true_ori):
        """
        Compute the lDDT (local distance difference test) score for the ligand–receptor interface,
        assuming the inputs are NumPy arrays.

        Args:
            ligand_coors_pred (np.ndarray): Predicted ligand coordinates of shape (n_res_lig, 3).
            receptor_coors_pred (np.ndarray): Predicted receptor coordinates of shape (n_res_rec, 3).
            ligand_coors_true_ori (np.ndarray): Ground truth ligand coordinates of shape (n_res_lig, 3).
            receptor_coors_true_ori (np.ndarray): Ground truth receptor coordinates of shape (n_res_rec, 3).

        Returns:
            float: The computed lDDT score.
        """
        # Compute pairwise distances between true ligand and receptor coordinates.
        # The difference has shape (n_res_rec, n_res_lig, 3)
        orig_diff = ligand_coors_true_ori[None, ...] - receptor_coors_true_ori[:, None, ...]
        orig_ca_lig_cross_distances = np.linalg.norm(orig_diff, axis=-1)
        
        # Compute pairwise distances for the predicted coordinates.
        pred_diff = ligand_coors_pred[None, ...] - receptor_coors_pred[:, None, ...]
        ca_lig_cross_distances = np.linalg.norm(pred_diff, axis=-1)
        
        # Compute the absolute difference between the true and predicted cross distances.
        ca_lig_cross_distances_diff = np.abs(orig_ca_lig_cross_distances - ca_lig_cross_distances)
        
        # Create a cutoff mask: only distances less than 15 Å count.
        cutoff_mask = (orig_ca_lig_cross_distances < 15.0).astype(np.float32)
        
        # Compute score based on distance difference thresholds.
        # The thresholds are: <1, <2, <4, and <8, each contributing 0.25.
        score = 0.25 * (
            (ca_lig_cross_distances_diff < 1.0).astype(np.float32)
            + (ca_lig_cross_distances_diff < 2.0).astype(np.float32)
            + (ca_lig_cross_distances_diff < 4.0).astype(np.float32)
            + (ca_lig_cross_distances_diff < 8.0).astype(np.float32)
        )
        
        # Compute the final lDDT score only over the valid (cutoff) distances.
        if cutoff_mask.sum() > 0:
            ca_lig_cross_lddt = (score * cutoff_mask).sum() / cutoff_mask.sum()
        else:
            print(f"{self.names[-1]} has no valid ca_lig_cross_distances under 15.0")
            ca_lig_cross_lddt = 0.0

        # Save the computed score.
        self.true_lddt_list.append(ca_lig_cross_lddt)
        
        return ca_lig_cross_lddt
        


    def update_interface_rmsd_recovery(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, receptor_coors_u, ligand_coors_u, cutoff=8.):
        
        ligand_receptor_distance = spa.distance.cdist(ligand_coors_true, receptor_coors_true)
        positive_tuple = np.where(ligand_receptor_distance < cutoff)

        if len(positive_tuple[0]) == 0:
            print(f"There is no interface residue (cutoff: {cutoff}) in the complex {self.names[-1]}. Please check the input data.")
        
        active_ligand = np.unique(positive_tuple[0])
        active_receptor = np.unique(positive_tuple[1])
        
        ligand_coors_pred = ligand_coors_pred[active_ligand, :]
        ligand_coors_true = ligand_coors_true[active_ligand, :]
        ligand_coors_u = ligand_coors_u[active_ligand, :]

        receptor_coors_pred = receptor_coors_pred[active_receptor, :]
        receptor_coors_true = receptor_coors_true[active_receptor, :]
        receptor_coors_u = receptor_coors_u[active_receptor, :]

    
        pred_to_b_ligand = compute_rmsd_align(ligand_coors_pred, ligand_coors_true)
        pred_to_b_receptor = compute_rmsd_align(receptor_coors_pred, receptor_coors_true)

        u_to_true_ligand = compute_rmsd_align(ligand_coors_u, ligand_coors_true)
        u_to_true_receptor = compute_rmsd_align(receptor_coors_u, receptor_coors_true)

        recovery_rmsd_prec_ligand = 1 - pred_to_b_ligand/u_to_true_ligand
        recovery_rmsd_prec_receptor = 1 - pred_to_b_receptor/u_to_true_receptor

        self.recovery_interface_rmsd_prec_ligand_list.append(recovery_rmsd_prec_ligand)
        self.recovery_interface_rmsd_prec_receptor_list.append(recovery_rmsd_prec_receptor)

        self.recovery_interface_rmsd_prec_list.append((recovery_rmsd_prec_ligand + recovery_rmsd_prec_receptor)/2)

        return (recovery_rmsd_prec_ligand + recovery_rmsd_prec_receptor)/2

        
        

    def update_complex_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, self_list=None):
        complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors_pred), axis=0)
        complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors_true), axis=0)

        R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

        complex_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)
        if self_list is None:
            self.complex_rmsd_list.append(complex_rmsd)
        else:
            self_list.append(complex_rmsd)

        return complex_rmsd

    def update_ind_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true):

        ligand_rmsd = compute_rmsd(ligand_coors_pred, ligand_coors_true)
        receptor_rmsd = compute_rmsd(receptor_coors_pred, receptor_coors_true)
        

        
        self.ind_rmsd_list.append([ligand_rmsd, receptor_rmsd])
        

        return [ligand_rmsd, receptor_rmsd]

    def update_ligand_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true):
        
        R,t = rigid_transform_Kabsch_3D(receptor_coors_pred.T, receptor_coors_true.T)

    
        ligand_coors_pred_aligned = (R @ ligand_coors_pred.T + t).T

        ligand_rmsd = compute_rmsd(ligand_coors_pred_aligned, ligand_coors_true)
        self.ligand_rmsd_list.append(ligand_rmsd)
        
        return ligand_rmsd


    def update_interface_rmsd(self, ligand_coors_pred, ligand_coors_true, receptor_coors_pred, receptor_coors_true, receptor_coors_true_ori=None, ligand_coors_true_ori=None, cutoff=8.0):
        if receptor_coors_true_ori is None: # if None, then use the coors_true
            receptor_coors_true_ori = receptor_coors_true
            ligand_coors_true_ori = ligand_coors_true
        
        ligand_receptor_distance = spa.distance.cdist(ligand_coors_true_ori, receptor_coors_true_ori)
        positive_tuple = np.where(ligand_receptor_distance < cutoff)

        if len(positive_tuple[0]) == 0:
            print(f"There is no interface residue in the complex {self.names[-1]}. Please check the input data.")
        
        active_ligand = np.unique(positive_tuple[0])
        active_receptor = np.unique(positive_tuple[1])
        
        ligand_coors_pred = ligand_coors_pred[active_ligand, :]
        ligand_coors_true = ligand_coors_true[active_ligand, :]

        receptor_coors_pred = receptor_coors_pred[active_receptor, :]
        receptor_coors_true = receptor_coors_true[active_receptor, :]
        

        complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors_pred), axis=0)
        complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors_true), axis=0)

        R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

        interface_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)
        self.interface_rmsd_list.append(interface_rmsd)

        return interface_rmsd

    def summarize(self, verbose=True):
        ligand_rmsd_summarized = get_rmsd_summary(self.ligand_rmsd_list) if self.ligand_rmsd_list != [] else None
        complex_rmsd_summarized = get_rmsd_summary(self.complex_rmsd_list)
        interface_rmsd_summarized = get_rmsd_summary(self.interface_rmsd_list)

        if self.docking_quality_list != []:
            docking_quality_summarized = get_quality_summary(self.docking_quality_list)

            if verbose:
                print(f'docking_quality_summarized: {docking_quality_summarized}')

        if self.recovery_rmsd_prec_list != []:
            recovery_rmsd_prec_summarized = get_rmsd_summary(self.recovery_rmsd_prec_list, percentage=True)
            recovery_interface_rmsd_prec_summarized = get_rmsd_summary(self.recovery_interface_rmsd_prec_list, percentage=True)

            if verbose:
                print(f'recovery_rmsd_prec_summarized: {recovery_rmsd_prec_summarized}')
                print(f'recovery_interface_rmsd_prec_summarized: {recovery_interface_rmsd_prec_summarized}')

        if verbose:
            print(f'ligand_rmsd_summarized: {ligand_rmsd_summarized}')
            print(f'complex_rmsd_summarized: {complex_rmsd_summarized}')
            print(f'interface_rmsd_summarized: {interface_rmsd_summarized}')

        return ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized, recovery_rmsd_prec_summarized if self.recovery_rmsd_prec_list != [] else None, recovery_interface_rmsd_prec_summarized if self.recovery_interface_rmsd_prec_list != [] else None

    def pretty_print(self, lrmsd, crmsd, irmsd, rrmsd=None):
        num_test_files = len(self.complex_rmsd_list)
        print(f'Number of samples:\t\t{num_test_files}')
        print()
        print(f"Ligand RMSD median/mean:\t{lrmsd['median']:.3}/{lrmsd['mean']:.3} ± {lrmsd['std']:.3}")
        print(f"Complex RMSD median/mean:\t{crmsd['median']:.3}/{crmsd['mean']:.3} ± {crmsd['std']:.3}")
        print(f"Interface RMSD median/mean:\t{irmsd['median']:.3}/{irmsd['mean']:.3} ± {irmsd['std']:.3}")
        if rrmsd is not None:
            print(f"recovery RMSD percentage median/mean:\t{rrmsd['median']:.3}/{rrmsd['mean']:.3} ± {rrmsd['std']:.3}")
        print()
        print(f"Ligand lt1/lt2/lt5/lt10:\t{lrmsd['lt1']:.3}%/{lrmsd['lt2']:.3}%/{lrmsd['lt5']:.3}%/{lrmsd['lt10']:.3}%")
        print(f"Complex lt1/lt2/lt5/lt10:\t{crmsd['lt1']:.3}%/{crmsd['lt2']:.3}%/{crmsd['lt5']:.3}%/{crmsd['lt10']:.3}%")
        print(f"Interface lt1/lt2/lt5/lt10:\t{irmsd['lt1']:.3}%/{irmsd['lt2']:.3}%/{irmsd['lt5']:.3}%/{irmsd['lt10']:.3}%")
        if rrmsd is not None:
            print(f"recovery RMSD percentage lt1/lt2/lt5/lt10:\t{rrmsd['lt1']:.3}%/{rrmsd['lt2']:.3}%/{rrmsd['lt5']:.3}%/{rrmsd['lt10']:.3}%")

# ############### tr_only ###############
# # [nm]
# def evaluate_all_rmsds(data_list, samples_list):
#     """
#         Evaluate sampled pose vs. ground truth
#     """
#     DEBUG = True

#     meter = RMSDComputer()
#     assert len(data_list) == len(samples_list)
#     for true_graph, pred_graph in zip(data_list, samples_list):

#         # [choice]
#         # Ending point: 1. opt_bound 2. ori_bound 
#         # !!!!! #
#         # 1. opt_bound
#         # true_lig_xyz = true_graph["ligand"].pos_b
#         # true_rec_xyz = true_graph["receptor"].pos_b

#         # 2. ori_bound
#         true_lig_xyz = true_graph["ligand"].pos_b_ori
#         true_rec_xyz = true_graph["receptor"].pos_b_ori
#         # !!!!! #



#         true_ori_rec_xyz = true_graph["receptor"].pos_b_ori
#         true_ori_lig_xyz = true_graph["ligand"].pos_b_ori

#         # mask
#         if 'mask' in true_graph["ligand"]:
#             lig_xyz_u = true_graph["ligand"].pos_u[true_graph["ligand"].mask]
#             rec_xyz_u = true_graph["receptor"].pos_u[true_graph["receptor"].mask]
            
#         else:
#             lig_xyz_u = true_graph["ligand"].pos_u
#             rec_xyz_u = true_graph["receptor"].pos_u
            

#         # [choice]
#         # predition based on 1. unbound 2. bound
#         # !!!!! #
        
#         # 1. predition is unbound
#         # mask
#         # if 'mask' in true_graph["ligand"]:
#         #     pred_lig_xyz = pred_graph["ligand"].pos[true_graph["ligand"].mask]
#         #     pred_rec_xyz = pred_graph["receptor"].pos[true_graph["receptor"].mask]
#         # else:
#         #     pred_lig_xyz = pred_graph["ligand"].pos
#         #     pred_rec_xyz = pred_graph["receptor"].pos
        

#         # # 2. predition is bound
#         # mask
#         if 'mask' in true_graph["ligand"]:
#             pred_lig_xyz = pred_graph["ligand"].pos
#             pred_rec_xyz = pred_graph["receptor"].pos
#         else:
#             pred_lig_xyz = pred_graph["ligand"].pos
#             pred_rec_xyz = pred_graph["receptor"].pos
        
#         # !!!!! #

#         if pred_graph["mirrored"]:
#             true_lig_xyz, true_rec_xyz = true_rec_xyz, true_lig_xyz
#             lig_xyz_u, rec_xyz_u = rec_xyz_u, lig_xyz_u
#             assert true_rec_xyz.shape == pred_rec_xyz.shape

#         assert true_lig_xyz.shape == pred_lig_xyz.shape

#         if not pred_graph["mirrored"]:
#             # assert (true_rec_xyz == pred_graph["receptor"].pos).all()
#             # assert (true_rec_xyz.shape == pred_graph["receptor"].pos.shape)
#             assert (true_rec_xyz.shape == pred_rec_xyz.shape)

#         try:
#             complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec = meter.update_all_rmsd(pred_lig_xyz.numpy(), true_lig_xyz.numpy(), pred_rec_xyz.numpy(), true_rec_xyz.numpy(), ligand_coors_u=lig_xyz_u.numpy(), receptor_coors_u=rec_xyz_u.numpy(), receptor_coors_true_ori=true_ori_rec_xyz.numpy(), ligand_coors_true_ori=true_ori_lig_xyz.numpy())
#         except Exception as e:
#             print(f"Error in {true_graph.name}: {e}")
#             print(f"true_lig_xyz: {true_lig_xyz.shape}, pred_lig_xyz: {pred_lig_xyz.shape}, true_rec_xyz: {true_rec_xyz.shape}, pred_rec_xyz: {pred_rec_xyz.shape}, lig_xyz_u: {lig_xyz_u.shape}, rec_xyz_u: {rec_xyz_u.shape}")
#             print(pred_graph.name)

#     return meter

# # [nm]
# def evaluate_all_ori_rmsds(data_list):
#     """
#         Evaluate unbound vs. bound
#     """
#     DEBUG = True

#     meter = RMSDComputer()
#     for true_graph in data_list:
#         # #
#         # if 'p7/1p7o.pdb1_3.dill' == true_graph.name:
#         #     continue

#         # [choice]
#         # Ending point:  1. opt_bound 2. ori_bound

#         # !!!!! #
#         # 1. opt_bound
#         # true_lig_xyz = true_graph["ligand"].pos_b
#         # true_rec_xyz = true_graph["receptor"].pos_b

#         # 2. ori_bound
#         true_lig_xyz = true_graph["ligand"].pos_b_ori
#         true_rec_xyz = true_graph["receptor"].pos_b_ori
#         # !!!!! #

#         true_ori_rec_xyz = true_graph["receptor"].pos_b_ori
#         true_ori_lig_xyz = true_graph["ligand"].pos_b_ori


#         if 'mask' in true_graph["ligand"]:
#             lig_xyz_u = true_graph["ligand"].pos_u[true_graph["ligand"].mask]
#             rec_xyz_u = true_graph["receptor"].pos_u[true_graph["receptor"].mask]
#         else:
#             lig_xyz_u = true_graph["ligand"].pos_u
#             rec_xyz_u = true_graph["receptor"].pos_u

#         # [choice]
#         # Starting point:  1. unbound 2. unbound sampled starting point 3. bound sampled starting point


#         # !!!!! #
#         # mask
            
#         # # 1. starting point is unbound
#         # pred_lig_xyz = lig_xyz_u
#         # pred_rec_xyz = rec_xyz_u
        

#         # 2. starting point is (tranfed) unbound
#         # if 'mask' in true_graph["ligand"]:
#         #     pred_lig_xyz = true_graph["ligand"].pos[true_graph["ligand"].mask]
#         #     pred_rec_xyz = true_graph["receptor"].pos[true_graph["receptor"].mask]
#         # else:
#         #     pred_lig_xyz = true_graph["ligand"].pos
#         #     pred_rec_xyz = true_graph["receptor"].pos

#         # # 3. starting point is (tranfed) bound
#         pred_lig_xyz = true_graph["ligand"].pos
#         pred_rec_xyz = true_graph["receptor"].pos

#         # !!!!! #
        
    


#         # if true_graph["mirrored"]:
#         #     true_lig_xyz, true_rec_xyz = true_rec_xyz, true_lig_xyz
#         #     lig_xyz_u, rec_xyz_u = rec_xyz_u, lig_xyz_u
            

#         # if not true_graph["mirrored"]:
#         #     # assert (true_rec_xyz == pred_graph["receptor"].pos).all()
#         #     # assert (true_rec_xyz.shape == pred_graph["receptor"].pos.shape)
            
#         if DEBUG:
#             print(true_graph.name)

#         complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec = meter.update_all_rmsd(pred_lig_xyz.numpy(), true_lig_xyz.numpy(), pred_rec_xyz.numpy(), true_rec_xyz.numpy(), ligand_coors_u=lig_xyz_u.numpy(), receptor_coors_u=rec_xyz_u.numpy(), receptor_coors_true_ori=true_ori_rec_xyz.numpy(), ligand_coors_true_ori=true_ori_lig_xyz.numpy())

#     return meter
############### tr_only ###############
            


############### flex_only ###############            
# # [nm]
# def evaluate_all_rmsds(data_list, samples_list):
#     """
#         Evaluate sampled pose vs. ground truth
#     """
#     DEBUG = True

#     meter = RMSDComputer()
#     assert len(data_list) == len(samples_list)
#     for true_graph, pred_graph in zip(data_list, samples_list):

#         # [choice]
#         # Ending point: 1. opt_bound 2. ori_bound 
#         # !!!!! #
#         # 1. opt_bound
#         true_lig_xyz = true_graph["ligand"].pos_b
#         true_rec_xyz = true_graph["receptor"].pos_b

#         # 2. ori_bound
#         # true_lig_xyz = true_graph["ligand"].pos_b_ori
#         # true_rec_xyz = true_graph["receptor"].pos_b_ori
#         # !!!!! #



#         true_ori_rec_xyz = true_graph["receptor"].pos_b_ori
#         true_ori_lig_xyz = true_graph["ligand"].pos_b_ori

#         # mask
#         if 'mask' in true_graph["ligand"]:
#             lig_xyz_u = true_graph["ligand"].pos_u[true_graph["ligand"].mask]
#             rec_xyz_u = true_graph["receptor"].pos_u[true_graph["receptor"].mask]
            
#         else:
#             lig_xyz_u = true_graph["ligand"].pos_u
#             rec_xyz_u = true_graph["receptor"].pos_u
            

#         # [choice]
#         # predition based on 1. unbound 2. bound
#         # !!!!! #
        
#         # 1. predition is unbound
#         # mask
#         if 'mask' in true_graph["ligand"]:
#             pred_lig_xyz = pred_graph["ligand"].pos[true_graph["ligand"].mask]
#             pred_rec_xyz = pred_graph["receptor"].pos[true_graph["receptor"].mask]
#         else:
#             pred_lig_xyz = pred_graph["ligand"].pos
#             pred_rec_xyz = pred_graph["receptor"].pos
        

#         # # 2. predition is bound
#         # mask
#         # if 'mask' in true_graph["ligand"]:
#         #     pred_lig_xyz = pred_graph["ligand"].pos
#         #     pred_rec_xyz = pred_graph["receptor"].pos
#         # else:
#         #     pred_lig_xyz = pred_graph["ligand"].pos
#         #     pred_rec_xyz = pred_graph["receptor"].pos
        
#         # !!!!! #

#         if pred_graph["mirrored"]:
#             true_lig_xyz, true_rec_xyz = true_rec_xyz, true_lig_xyz
#             lig_xyz_u, rec_xyz_u = rec_xyz_u, lig_xyz_u
#             assert true_rec_xyz.shape == pred_rec_xyz.shape

#         assert true_lig_xyz.shape == pred_lig_xyz.shape

#         if not pred_graph["mirrored"]:
#             # assert (true_rec_xyz == pred_graph["receptor"].pos).all()
#             # assert (true_rec_xyz.shape == pred_graph["receptor"].pos.shape)
#             assert (true_rec_xyz.shape == pred_rec_xyz.shape)

#         try:
#             complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec = meter.update_all_rmsd(pred_lig_xyz.numpy(), true_lig_xyz.numpy(), pred_rec_xyz.numpy(), true_rec_xyz.numpy(), ligand_coors_u=lig_xyz_u.numpy(), receptor_coors_u=rec_xyz_u.numpy(), receptor_coors_true_ori=true_ori_rec_xyz.numpy(), ligand_coors_true_ori=true_ori_lig_xyz.numpy(), name = true_graph.name)
#         except Exception as e:
#             print(f"Error in {true_graph.name}: {e}")
#             print(f"true_lig_xyz: {true_lig_xyz.shape}, pred_lig_xyz: {pred_lig_xyz.shape}, true_rec_xyz: {true_rec_xyz.shape}, pred_rec_xyz: {pred_rec_xyz.shape}, lig_xyz_u: {lig_xyz_u.shape}, rec_xyz_u: {rec_xyz_u.shape}")
#             print(pred_graph.name)

#     return meter



# [nm]
def evaluate_all_rmsds(samples_list):
    """
        Evaluate sampled pose vs. ground truth
    """
    DEBUG = True
    # print("label is optimally fitted bound")
    print("label is original bound")

    meter = RMSDComputer()
    for pred_graph in samples_list:

        # [choice]
        # Ending point: 1. opt_bound 2. ori_bound 
        # !!!!! #
        # 1. opt_bound
        # true_lig_xyz = pred_graph["ligand"].pos_b
        # true_rec_xyz = pred_graph["receptor"].pos_b

        # 2. ori_bound 
        true_lig_xyz = pred_graph["ligand"].pos_b_ori
        true_rec_xyz = pred_graph["receptor"].pos_b_ori
        # !!!!! #



        true_ori_rec_xyz = pred_graph["receptor"].pos_b_ori
        true_ori_lig_xyz = pred_graph["ligand"].pos_b_ori

        # mask
        if 'mask' in pred_graph["ligand"]:
            lig_xyz_u = pred_graph["ligand"].pos_u[pred_graph["ligand"].mask]
            rec_xyz_u = pred_graph["receptor"].pos_u[pred_graph["receptor"].mask]
            
        else:
            lig_xyz_u = pred_graph["ligand"].pos_u
            rec_xyz_u = pred_graph["receptor"].pos_u
            

        # [choice]
        # predition based on 1. unbound 2. bound
        # !!!!! #
        
        # 1. predition is unbound
        # mask
        if 'mask' in pred_graph["ligand"]:
            # pred_lig_xyz = pred_graph["ligand"].pos[pred_graph["ligand"].mask]
            # pred_rec_xyz = pred_graph["receptor"].pos[pred_graph["receptor"].mask]
            pred_lig_xyz = pred_graph["ligand"].pos
            pred_rec_xyz = pred_graph["receptor"].pos
        else:
            pred_lig_xyz = pred_graph["ligand"].pos
            pred_rec_xyz = pred_graph["receptor"].pos
        

        # # 2. predition is bound
        # mask
        # if 'mask' in true_graph["ligand"]:
        #     pred_lig_xyz = pred_graph["ligand"].pos
        #     pred_rec_xyz = pred_graph["receptor"].pos
        # else:
        #     pred_lig_xyz = pred_graph["ligand"].pos
        #     pred_rec_xyz = pred_graph["receptor"].pos
        
        # !!!!! #

        # if pred_graph["mirrored"]:
        #     true_lig_xyz, true_rec_xyz = true_rec_xyz, true_lig_xyz
        #     lig_xyz_u, rec_xyz_u = rec_xyz_u, lig_xyz_u
        #     assert true_rec_xyz.shape == pred_rec_xyz.shape

        assert true_lig_xyz.shape == pred_lig_xyz.shape

        if not pred_graph["mirrored"]:
            # assert (true_rec_xyz == pred_graph["receptor"].pos).all()
            # assert (true_rec_xyz.shape == pred_graph["receptor"].pos.shape)
            assert (true_rec_xyz.shape == pred_rec_xyz.shape)

        try:
            complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec = meter.update_all_rmsd(pred_lig_xyz.numpy(), true_lig_xyz.numpy(), pred_rec_xyz.numpy(), true_rec_xyz.numpy(), ligand_coors_u=lig_xyz_u.numpy(), receptor_coors_u=rec_xyz_u.numpy(), receptor_coors_true_ori=true_ori_rec_xyz.numpy(), ligand_coors_true_ori=true_ori_lig_xyz.numpy(), name = pred_graph.name)

            recovery_rmsd_prec_ligand, recovery_rmsd_prec_receptor = meter.recovery_rmsd_prec_ligand_list[-1], meter.recovery_rmsd_prec_receptor_list[-1] 
            recovery_interface_rmsd_prec_ligand, recovery_interface_rmsd_prec_receptor = meter.recovery_interface_rmsd_prec_ligand_list[-1], meter.recovery_interface_rmsd_prec_receptor_list[-1]
            fnat = meter.fnat_list[-1]
            quality = meter.docking_quality_list[-1]

            name = pred_graph.name
            print("################")
            print(f"{name}: complex_rmsd: {complex_rmsd:.3f}, ligand_rmsd: {ligand_rmsd:.3f}, interface_rmsd: {interface_rmsd:.3f}")
            print(f"{name}: recovery_rmsd_prec_ligand: {recovery_rmsd_prec_ligand:.3f}, recovery_rmsd_prec_receptor: {recovery_rmsd_prec_receptor:.3f}")
            print(f"{name}: recovery_interface_rmsd_prec_ligand: {recovery_interface_rmsd_prec_ligand:.3f}, recovery_interface_rmsd_prec_receptor: {recovery_interface_rmsd_prec_receptor:.3f}") 
            print(f"{name}: fnat: {fnat:.3f}")
            print(f"{name}: quality: {quality}")
                  
        except Exception as e:
            print(f"Error in {pred_graph.name}: {e}")
            print(f"true_lig_xyz: {true_lig_xyz.shape}, pred_lig_xyz: {pred_lig_xyz.shape}, true_rec_xyz: {true_rec_xyz.shape}, pred_rec_xyz: {pred_rec_xyz.shape}, lig_xyz_u: {lig_xyz_u.shape}, rec_xyz_u: {rec_xyz_u.shape}")
            print(pred_graph.name)

    return meter

# [nm]
def evaluate_all_ori_rmsds(data_list):
    """
        Evaluate unbound vs. bound
    """
    DEBUG = True

    meter = RMSDComputer()
    for true_graph in data_list:
        # #
        # if 'p7/1p7o.pdb1_3.dill' == true_graph.name:
        #     continue

        # [choice]
        # Ending point:  1. opt_bound 2. ori_bound

        # !!!!! #
        # 1. opt_bound
        # true_lig_xyz = true_graph["ligand"].pos_b
        # true_rec_xyz = true_graph["receptor"].pos_b

        # 2. ori_bound
        true_lig_xyz = true_graph["ligand"].pos_b_ori
        true_rec_xyz = true_graph["receptor"].pos_b_ori
        # !!!!! #

        true_ori_rec_xyz = true_graph["receptor"].pos_b_ori
        true_ori_lig_xyz = true_graph["ligand"].pos_b_ori


        if 'mask' in true_graph["ligand"]:
            lig_xyz_u = true_graph["ligand"].pos_u[true_graph["ligand"].mask]
            rec_xyz_u = true_graph["receptor"].pos_u[true_graph["receptor"].mask]
        else:
            lig_xyz_u = true_graph["ligand"].pos_u
            rec_xyz_u = true_graph["receptor"].pos_u

        # [choice]
        # Starting point:  1. unbound 2. unbound sampled starting point 3. bound sampled starting point


        # !!!!! #
        # mask
            
        # # 1. starting point is unbound
        # pred_lig_xyz = lig_xyz_u
        # pred_rec_xyz = rec_xyz_u
        

        # 2. starting point is (tranfed) unbound
        if 'mask' in true_graph["ligand"]:
            # pred_lig_xyz = true_graph["ligand"].pos[true_graph["ligand"].mask]
            # pred_rec_xyz = true_graph["receptor"].pos[true_graph["receptor"].mask]
            pred_lig_xyz = true_graph["ligand"].pos
            pred_rec_xyz = true_graph["receptor"].pos
        
        else:
            pred_lig_xyz = true_graph["ligand"].pos
            pred_rec_xyz = true_graph["receptor"].pos

        # # 3. starting point is (tranfed) bound
        # pred_lig_xyz = true_graph["ligand"].pos
        # pred_rec_xyz = true_graph["receptor"].pos

        # !!!!! #
        
    


        # if true_graph["mirrored"]:
        #     true_lig_xyz, true_rec_xyz = true_rec_xyz, true_lig_xyz
        #     lig_xyz_u, rec_xyz_u = rec_xyz_u, lig_xyz_u
            

        # if not true_graph["mirrored"]:
        #     # assert (true_rec_xyz == pred_graph["receptor"].pos).all()
        #     # assert (true_rec_xyz.shape == pred_graph["receptor"].pos.shape)
            
        if DEBUG:
            print(true_graph.name)

        complex_rmsd, ligand_rmsd, interface_rmsd, recovery_rmsd_prec = meter.update_all_rmsd(pred_lig_xyz.numpy(), true_lig_xyz.numpy(), pred_rec_xyz.numpy(), true_rec_xyz.numpy(), ligand_coors_u=lig_xyz_u.numpy(), receptor_coors_u=rec_xyz_u.numpy(), receptor_coors_true_ori=true_ori_rec_xyz.numpy(), ligand_coors_true_ori=true_ori_lig_xyz.numpy(), name = true_graph.name)

    return meter

############### flex_only ###############