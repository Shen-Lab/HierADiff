# !!! [caution]

import os
import sys
import dill
import pickle
import random
import warnings
import itertools
from collections import defaultdict
import math
import csv

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.data import HeteroData

from scipy.spatial.transform import Rotation

import Bio
# These are just annoying :')
import Bio.PDB
warnings.filterwarnings("ignore",
    category=Bio.PDB.PDBExceptions.PDBConstructionWarning)
from Bio.Data.IUPACData import protein_letters_3to1

from utils import load_csv, printt
from utils import compute_rmsd

from prody import *
from evaluation.compute_rmsd_nm import rigid_transform_Kabsch_3D


# -------- DATA LOADING -------
DATA_CACHE_VERSION = "v2"

class Loader:
    """
        Raw data loader with shared utilities
        @param (bool) data  default=True to load complexes, not poses
    """
    def __init__(self, args, data=True):
        self.args = args
        if data:
            # self.root = args.data_path = 'datasets/DIPS/pairs_pruned'
            self.root = args.data_path
            # self.data_file = args.data_file = 'datasets/DIPS/data_file.csv'
            self.data_file = args.data_file
        else:
            self.root = args.pose_path
            self.data_file = args.pose_file
        
        # read in file paths
        self.data_to_load = load_csv(self.data_file)

        self.data_cache = self.data_file.replace(".csv", f"_cache_{DATA_CACHE_VERSION}.pkl")
        # cache for post-processed data, optional
        self.graph_cache = self.data_file.replace(".csv",
                f"_graph_{args.resolution}.pkl")
        self.esm_cache = self.data_file.replace(".csv", f"_esm.pkl")
        #if args.use_orientation_features:
        #    self.data_cache = self.data_cache.replace(".pkl", "_orientation.pkl")
        #    self.graph_cache = self.graph_cache.replace(".pkl", "_orientation.pkl")
        #    self.esm_cache = self.esm_cache.replace(".pkl", "_orientation.pkl")
        if args.debug:
            self.data_cache = self.data_cache.replace(".pkl", "_debug.pkl")
            self.graph_cache = self.graph_cache.replace(".pkl", "_debug.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_debug.pkl")
        # biopython PDB parser
        self.parser = Bio.PDB.PDBParser()

    def read_files(self):
        raise Exception("implement me")

    @staticmethod
    def add_values_to_dict(dictionary, **values):
        for key, value in values.items():
            dictionary[key].append(value)

    def parse_pdb(self, fp, models=None, chains=None):
        """
            Parse PDB file via Biopython
            (instead of my hacky implementation based on the spec)
            @return  (residue sequence, coordinates, atom sequence)
        """
        all_res, all_atom, all_pos = [], [], []
        visualization_values = defaultdict(list)

        structure = self.parser.get_structure(fp, fp)  # name, path
        # (optional) select subset of models
        if models is not None:
            models = [structure[m] for m in models]
        else:
            models = structure
        for model in models:
            # (optional) select subset of chains
            if chains is not None:
                chains = [model[c] for c in chains if c in model]
            else:
                chains = model
            # retrieve all atoms in desired chains
            for chain in chains:
                res = chain.get_residues()
                for res in chain:
                    # ignore HETATM records
                    if res.id[0] != " ":
                        continue
                    for atom in res:
                        all_res.append(res.get_resname())
                        all_pos.append(torch.from_numpy(atom.get_coord()))
                        all_atom.append((atom.get_name(), atom.element))
                        self.add_values_to_dict(visualization_values,
                                                chain=repr(chain).replace("<Chain id=", "").replace(">", ""),
                                                residue=res.get_segid(),
                                                atom_name=atom.get_name(),
                                                element=atom.element)
        all_pos = torch.stack(all_pos, dim=0)
        visualization_values["resname"] = all_res

        return all_res, all_atom, all_pos, visualization_values

    def _to_dict(self, pdb, p1, p2):
        item = {
            "pdb_id": pdb,
            "receptor": p1,
            "ligand": p2
        }
        return item
    
    # [nm]
    def _to_dict_nm(self, pdb, p1_b, p2_b, p1_u, p2_u):
        item = {
            "pdb_id": pdb,
            "receptor_b": p1_b,
            "ligand_b": p2_b,
            "receptor_u": p1_u,
            "ligand_u": p2_u,
        }
        return item

    # [nm]
    def convert_pdb(self, all_res, all_atom, all_pos, all_visualization_values):
        """
            Unify PDB representation across different dataset formats.
            Given all residues, coordinates, and atoms, select subset
            of all atoms to keep.

            @return tuple(list, torch.Tensor) seq, pos
        """
        if self.args.resolution == "atom":
            atoms_to_keep = None
        elif self.args.resolution == "backbone":
            atoms_to_keep = ["N", "CA", "O", "CB"]
        elif self.args.resolution == "residue":
            atoms_to_keep = ["CA"]
        else:
            raise Exception(f"invalid resolution {self.args.resolution}")

        if atoms_to_keep is not None:
            to_keep, atoms = [], []
            for i,a in enumerate(all_atom):
                if a[0] in atoms_to_keep:  # atom name
                    to_keep.append(i)
                    atoms.append(a[1])  # element symbol
            seq = [all_res[i] for i in to_keep]
            visualization_values = {key: [values[i] for i in to_keep] for key, values in all_visualization_values.items()}
            pos = all_pos[torch.tensor(to_keep)]
        else:
            atoms = all_atom
            seq = all_res
            pos = all_pos
            visualization_values = all_visualization_values
        assert pos.shape[0] == len(seq) == len(atoms)
        return atoms, seq, pos, visualization_values

    def extract_coords(self, all_res, all_atom, all_pos, all_visualization_values):
        """
            Extract coordinates of CA, N, and C atoms that are needed 
            to compute orientation features, as proposed in EquiDock

            @return c_alpha_coords, n_coords, c_coords (all torch.Tensor)
        """
        c_alpha_coords = []
        n_coords = []
        c_coords = []
        for i, a in enumerate(all_atom):
            if a[0] == "CA":
                c_alpha_coords.append(all_pos[i])
            if a[0] == "N":
                n_coords.append(all_pos[i])
            if a[0] == "C":
                c_coords.append(all_pos[i])

        c_alpha_coords = torch.stack(c_alpha_coords)
        n_coords = torch.stack(n_coords)
        c_coords = torch.stack(c_coords)
        
        assert c_alpha_coords.shape == n_coords.shape == c_coords.shape

        return c_alpha_coords, n_coords, c_coords

    def compute_orientation_vectors(self, c_alpha_coords, n_coords, c_coords):
        """
            Extract 3D coordinates and n_i, u_i, v_i vectors of representative residues
        """
        num_residues = c_alpha_coords.shape[0]
        
        n_i_list = []
        u_i_list = []
        v_i_list = []
        for i in range(num_residues):
            n_coord = n_coords[i]
            c_alpha_coord = c_alpha_coords[i]
            c_coord = c_coords[i]
            u_i = (n_coord - c_alpha_coord) / torch.linalg.vector_norm(n_coord - c_alpha_coord)
            t_i = (c_coord - c_alpha_coord) / torch.linalg.vector_norm(c_coord - c_alpha_coord)
            n_i = torch.linalg.cross(u_i, t_i) / torch.linalg.vector_norm(torch.linalg.cross(u_i, t_i))
            v_i = torch.linalg.cross(n_i, u_i)
            assert (torch.abs(torch.linalg.vector_norm(v_i) - 1.) < 1e-5), "v_i norm is not 1"
            n_i_list.append(n_i)
            u_i_list.append(u_i)
            v_i_list.append(v_i)

        n_i_feat = torch.stack(n_i_list)
        u_i_feat = torch.stack(u_i_list)
        v_i_feat = torch.stack(v_i_list)
        
        assert n_i_feat.shape == u_i_feat.shape == v_i_feat.shape
        
        return n_i_feat, u_i_feat, v_i_feat

    def process_data(self, data=None, args=None):
        """
            Tokenize, etc.

            AND

            Filter out the complex that has any r/l seq len less than self.args.filter_len_r_or_l
        """
        # check if cache exists
        # [Modified]
        # ???

        if self.args.data_process and self.args.totalInd > 1:
            boolean_value = (not self.flag) and (not self.args.recache) and (os.path.exists(self.graph_cache  + str(self.args.ind)))
            # save_path = self.data_cache + str(self.args.ind)
        else:
            boolean_value = (not self.flag) and (not self.args.recache) and (os.path.exists(self.graph_cache))
            # save_path = self.data_cache

        # if (not self.flag) and (not self.args.recache) and (os.path.exists(self.graph_cache)):
        # if (not self.flag) and (not self.args.recache) and (os.path.exists(self.graph_cache  + str(self.args.ind))):
        # if not args.no_graph_cache and os.path.exists(self.graph_cache):
        if boolean_value:
            

            if self.args.data_process and self.args.totalInd > 1:
                # check if the file exists (priority)
                if os.path.exists(self.graph_cache  + str(self.args.ind)):
                    # load the file
                    with open(self.graph_cache  + str(self.args.ind), "rb") as f:
                        data, data_params = pickle.load(f)
                        printt(f"Loaded processed data from cache for ind {self.args.ind}")
                    return data, data_params
                else:
                    with open(self.graph_cache, "rb") as f:
                        data, data_params = pickle.load(f)
                        printt("Loaded processed data from cache")
                    
                    # distribute the data according to the totalInd and the ind
                    len_data = len(data)
                    # the data will be divided into totalInd parts
                    # the ind-th part will be used
                    step = len_data // self.args.totalInd + 1
                    start = self.args.ind * step
                    end = (self.args.ind + 1) * step

                    sort_data = sorted(data)
                    new_data = {}
                    for i in range(start, min(end, len_data)):
                        new_data[sort_data[i]] = data[sort_data[i]]
                    
                    data = new_data # data_params is always {}
                    printt(f"Processed data for ind {self.args.ind}")

                    return data, data_params
                
            # ???
                
            if self.args.data_process and self.args.totalInd > 1:
                save_path = self.graph_cache  + str(self.args.ind)
            else:
                save_path = self.graph_cache
            # with open(self.graph_cache, "rb") as f:
            # with open(self.graph_cache  + str(self.args.ind), "rb") as f:
            with open(save_path, "rb") as f:
                data, data_params = pickle.load(f)
                printt("Loaded processed data from cache")
                return data, data_params

        # no data parameters until now
        data_params = {}

        #### select subset of residues that match desired
        # resolution (e.g. residue-level vs. atom-level)
        
        for path, item in tqdm(data.items(), desc='convert_pdb; in process_data'): # path e.g. 'ke/1kee.pdb1_6.dill'
            
             


            item["visualization_values"] = {}
            for k in ["receptor_b", "ligand_b", "receptor_u", "ligand_u"]:
                # >>> in what case would this be false?
                if k in item:
                    # atoms, seq, pos, visualization_values
                    subset = self.convert_pdb(*item[k])
                    item[f"{k}_atom"] = subset[0]
                    item[f"{k}_seq"] = subset[1]
                    item[f"{k}_xyz"] = subset[2]
                    item["visualization_values"][k] = subset[3]


                    # bb_atoms
                    try:
                        c_alpha_coords, n_coords, c_coords = self.extract_coords(*item[k])
                    
                        c_alpha_coords = c_alpha_coords.unsqueeze(1)
                        n_coords = n_coords.unsqueeze(1)
                        c_coords = c_coords.unsqueeze(1)

                        all_bb_coords = torch.cat([n_coords, c_alpha_coords, c_coords], dim=1)

                        item[f"{k}_bb_coords"] = all_bb_coords
                    except Exception as e:
                        print(f"Error in extract_coords: {e}")
                        item[f"{k}_bb_coords"] = None
                        continue


                    # for orientation features we need coordinates of CA, N, and C atoms
                    # These coordinates define the local coordinate system of each residue
                    if args.use_orientation_features:
                        n_i_feat, u_i_feat, v_i_feat = self.compute_orientation_vectors(c_alpha_coords, n_coords, c_coords)
                        item[f"{k}_n_i_feat"] = n_i_feat
                        item[f"{k}_u_i_feat"] = u_i_feat
                        item[f"{k}_v_i_feat"] = v_i_feat
                    
                    # {path: 
                    #     {
                    #     "pdb_id": pdb/path, # eg. 'ke/1kee.pdb1_6.dill'
                    #     "receptor_b": p1_b,
                    #     "ligand_b": p2_b,
                    #     "receptor_u": p1_u,
                    #     "ligand_u": p2_u,
                    #     'path': pdb/path, 
                    #     'split': 
                    #     }
                    # }

                    # item[f"{k}_atom"] = subset[0]
                    # item[f"{k}_seq"] = subset[1]
                    # item[f"{k}_xyz"] = subset[2]
                    # item["visualization_values"][k] = subset[3]
                    # item[f"{k}_n_i_feat"] = n_i_feat
                    # item[f"{k}_u_i_feat"] = u_i_feat
                    # item[f"{k}_v_i_feat"] = v_i_feat
            

        #### convert to HeteroData graph objects
        new_data = {}
        ct = 0
        for key, item in tqdm(data.items(), desc='to_graph; in process_data'):
            print(key)

            if item[f"receptor_b_bb_coords"] is None or item[f"ligand_b_bb_coords"] is None or item[f"receptor_u_bb_coords"] is None or item[f"ligand_u_bb_coords"] is None:
                print(f'skip {key} because one of the bb_coords is None.')
                ct += 1
                continue

            if self.args.model_type == 'confidence':
                # [mask]
                # self.mask = {'3o91.pdb1_0':{'r': list([True/False]), 'l': list([True/False])}, ...}
                item["receptor_mask"] = torch.tensor(self.mask[key[3:7]]["r"])
                item["ligand_mask"] = torch.tensor(self.mask[key[3:7]]["l"])
            else:
                # [mask]
                # self.mask = {'3o91.pdb1_0':{'r': list([True/False]), 'l': list([True/False])}, ...}
                item["receptor_mask"] = torch.tensor(self.mask[key[3:-5]]["r"])
                item["ligand_mask"] = torch.tensor(self.mask[key[3:-5]]["l"])

            


            rec = item["receptor_u_seq"]
            lig = item["ligand_u_seq"]

            # sum of mask

            len_rec_b = item["receptor_mask"].sum()

            len_lig_b = item["ligand_mask"].sum()

            if len_rec_b <= self.args.filter_len_r_or_l or len_lig_b <= self.args.filter_len_r_or_l:
                print(f'skip {key} because r or l is too small in the complex.')
                ct += 1
                continue

            try:
                item["graph"] = self.to_graph(item)
            except Exception as e:
                print(f"Error in to_graph: {e}")
                continue



            new_data[key] = item

        data = new_data
        
        print(f'skip {ct} complexes because either r or l is too small in those complexes.')
        


        if not args.no_graph_cache: # defaulted as False
            # ???
            if self.args.data_process and self.args.totalInd > 1:
                save_path = self.graph_cache  + str(self.args.ind)
            else:
                save_path = self.graph_cache

            # with open(self.graph_cache  + str(self.args.ind), "wb+") as f:
            # with open(self.graph_cache, "wb+") as f:
            with open(save_path, "wb+") as f:
                pickle.dump([data, data_params], f)

        # ???
        # if args.data_process == True
        # if there is a file with the subindex (ind) and the file will be used and loaded 
        # or the data will be created according to the totalInd and the ind by even distribution
        if self.args.data_process and self.args.totalInd > 1:
            # check if the file exists
            if os.path.exists(self.graph_cache  + str(self.args.ind)):
                # load the file
                with open(self.graph_cache  + str(self.args.ind), "rb") as f:
                    data, data_params = pickle.load(f)
                    printt(f"Loaded processed data from cache for ind {self.args.ind}")
                return data, data_params
            else:
                # distribute the data according to the totalInd and the ind
                len_data = len(data)
                # the data will be divided into totalInd parts
                # the ind-th part will be used
                step = len_data // self.args.totalInd + 1
                start = self.args.ind * step
                end = (self.args.ind + 1) * step

                sort_data = sorted(data)
                new_data = {}
                for i in range(start, min(end, len_data)):
                    new_data[sort_data[i]] = data[sort_data[i]]
                
                data = new_data # data_params is always {}
                printt(f"Processed data for ind {self.args.ind}")

                return data, data_params

        # data = {pdb: {'graph': HeteroData, f"{k}_atom": , "{k}_seq": , f"{k}_xyz": , "visualization_values": {"receptor": , "ligand": }}}
        # data_params = {"num_residues":, "tokenizer":}
        return data, data_params

    def process_embed(self, data, data_params):
        #### tokenize AFTER converting to graph
        if self.args.lm_embed_dim > 0:
            data = self.compute_embeddings(data)
            data_params["num_residues"] = 23  # <cls> <sep> <pad>
            printt("finished tokenizing residues with ESM")
        else:
            # tokenize residues for non-ESM
            tokenizer = tokenize(data.values(), "receptor_seq")
            tokenize(data.values(), "ligand_seq", tokenizer)
            self.esm_model = None
            data_params["num_residues"] = len(tokenizer)
            data_params["tokenizer"] = tokenizer
            printt("finished tokenizing residues")


        #### protein sequence tokenization
        # tokenize atoms
        atom_tokenizer = tokenize(data.values(), "receptor_u_atom")
        tokenize(data.values(), "ligand_u_atom", atom_tokenizer)
        tokenize(data.values(), "receptor_b_atom", atom_tokenizer)
        tokenize(data.values(), "ligand_b_atom", atom_tokenizer)
        data_params["atom_tokenizer"] = atom_tokenizer
        printt("finished tokenizing all inputs")

        return data, data_params

    def match_b_AND_calc_nm(self, data, item, key):
        '''
        Input:
            1. data: the heterodata graph object
            2. item: 
            3. key: "receptor"/"ligand"

        Return: data 
            including:
            1. data[key].b_nm
            2. data[key].u_nm
            3. data[key].cur_nm

            4. data[key].pos
            5. data[key].pos_u
            6. data[key].pos_b
        '''
        DEBUG = False
        # add NM node features


        #
        # unbound
        np_new_pos_u = item[f"{key}_u_xyz"].float().detach().numpy() # (n_res, 3)

        #
        # bound
        np_new_pos_b = item[f"{key}_b_xyz"].float().detach().numpy() # (n_res, 3)

        R,t = rigid_transform_Kabsch_3D(np_new_pos_u[item[f"{key}_mask"]].T, np_new_pos_b.T)
        # print("R:", R, "t:", t)

        # aligned pos_u
        np_new_pos_u_aligned = (R @ np_new_pos_u.T + t).T

        # !!!
        data[key].pos_u = torch.from_numpy(np_new_pos_u_aligned.astype(np.float32)) 
        

        # if DEBUG:
        #     print('np_new_x_l shape', np_new_x_l.shape)
            # return None

        # compute the hessian matrix, eigenvectors, and eigenvalues (including all normal modes)
        calphas = np_new_pos_u_aligned.astype(np.float32)[item[f"{key}_mask"]]
        
        anm = ANM()
        anm.buildHessian(calphas)
        hessian = anm.getHessian()
        hessian = torch.from_numpy(hessian).float()

        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        
        # Sort non-zero eigenvalues for stability
        sorted_indices = torch.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        data[key].H = hessian
        data[key].Lambda = eigenvalues
        data[key].P = eigenvectors

        

        # compute the first n_NM non-trivial normal modes
        calphas = np_new_pos_u_aligned.astype(np.float32)[item[f"{key}_mask"]]
        
        anm = ANM()
        anm.buildHessian(calphas)
        anm.calcModes(self.args.n_NM)
        eigvecs = anm.getEigvecs()
        eigvals = anm.getEigvals()

        # [DEBUG]
        # if DEBUG:
        # print('eigvecs shape', eigvecs.shape)
        # (n_res * 3, n_NM)

        # reshape to the shape (n_res, 3, n_NM)
        reshaped_np_eigvecs = eigvecs.reshape((-1,3,self.args.n_NM))

        # [DEBUG]
        # if DEBUG:
        #     print('reshaped_np_eigvecs shape', reshaped_np_eigvecs.shape)
        #     # (n_res, 3, n_NM)
        #     return None

        # (n_res, 3, n_NM)
        # !!!
        data[key].u_nm = torch.from_numpy(reshaped_np_eigvecs.astype(np.float32))
        if DEBUG:
            print('data[key].u_nm shape', data[key].u_nm.shape)

        # !!!
        data[key].eig_vals_u = torch.from_numpy(eigvals)


        # b_nm (n_res, 3, n_NM)
        np_new_pos_u_aligned_masked = np_new_pos_u_aligned[item[f"{key}_mask"]]

        calphas = np_new_pos_u_aligned_masked
        
        anm = ANM()
        anm.buildHessian(calphas)
        anm.calcModes(self.args.n_NM)
        eigvecs = anm.getEigvecs()
        eigvals = anm.getEigvals()

        # reshape to the shape (n_res, 3, n_NM)
        np_b_nm = eigvecs.reshape((-1,3,self.args.n_NM))

        # !!!
        data[key].b_nm = torch.from_numpy(np_b_nm.astype(np.float32))
        if DEBUG:
            print('data[key].b_nm shape', data[key].b_nm.shape)

        # !!!
        data[key].eig_vals_b = torch.from_numpy(eigvals)

        ############
        # # reshaped_np_eigvecs (n_res, 3, n_NM)
        # reshaped_np_eigvecs_b = reshaped_np_eigvecs[item[f"{key}_mask"]]
        # # (n_res, 3, n_NM) -> (n_res, n_NM, 3) -> (n_res*n_NM, 3) -> (3, n_res*n_NM) -> (3, n_res*n_NM) -> (n_res*n_NM, 3) -> (n_res, n_NM, 3)
        # np_to_transpose = (R @ (np.transpose(reshaped_np_eigvecs_b, (0, 2, 1)).reshape((-1, 3))).T).T.reshape((-1, self.args.n_NM,3))
        # np_b_nm = np.transpose(np_to_transpose, (0, 2, 1))
        # !!!
        # data[key].b_nm = torch.from_numpy(np_b_nm.astype(np.float32))
        # if DEBUG:
        #     print('data[key].b_nm shape', data[key].b_nm.shape)
        



        # # b (n*3, )
        # np_new_pos_diff = np_new_pos_b - np_new_pos_u_aligned[item[f"{key}_mask"]]
        # reshaped_np_new_pos_diff = np_new_pos_diff.reshape((-1,))
        # # A (n*3, n_NM)
        # reshaped_np_b_nm = np_b_nm.reshape((-1, self.args.n_NM))

        # # coeffs (n_NM, )
        # coeffs, residuals, rank, s = np.linalg.lstsq(reshaped_np_b_nm , reshaped_np_new_pos_diff, rcond=None)
        # data[key].b_to_u_nmCoeffs = -torch.from_numpy(coeffs.astype(np.float32))

        # # (n, 3)
        # np_pos_b_matched = (reshaped_np_b_nm@coeffs).reshape((-1, 3)) + np_new_pos_u_aligned[item[f"{key}_mask"]]
        #############

        #
        np_masked_centered_u = np_new_pos_u_aligned_masked - np_new_pos_u_aligned_masked.mean(0)
        np_centered_b_ori = np_new_pos_b - np_new_pos_b.mean(0) 

        # b (3n,)
        b = (np_masked_centered_u - np_centered_b_ori).reshape((-1,))
        # A (3n,n_NM)
        reshaped_np_b_nm = np_b_nm.reshape((-1, self.args.n_NM))

        #
        # b_to_u_nmCoeffs (n_NM,)
        np_coeffs, residuals, rank, s = np.linalg.lstsq(reshaped_np_b_nm, b, rcond=None)

        # !!!
        data[key].b_to_u_nmCoeffs = torch.from_numpy(np_coeffs.astype(np.float32))

        #
        # pos_b
        np_pos_b_matched = np_new_pos_u_aligned_masked - (np_b_nm@np_coeffs).reshape((-1, 3))


        # !!!
        data[key].pos_b = torch.from_numpy(np_pos_b_matched.astype(np.float32))
        # !!!
        data[key].pos = torch.from_numpy(np_new_pos_u_aligned.astype(np.float32)) # torch.from_numpy(np_pos_b_matched.astype(np.float32))
        if DEBUG:
            print('data[key].pos shape', data[key].pos.shape)
        # !!!
        data[key].cur_nm = None

        # !!!
        data[key].pos_b_ori = item[f"{key}_b_xyz"].float()

        return data


    # [nm]
    def to_graph(self, item):
        """
            Convert raw dictionary to PyTorch geometric object
        """
        data = HeteroData()
        data["name"] = item["path"]
        # retrieve position and compute kNN
        for key in ["receptor", "ligand"]:

            # [!!!!!!!!!!! mask]

            data = self.match_b_AND_calc_nm(data, item, key)

            # [!!!!!!!!!!! mask]
            # [TODO: manage data[key].x in the dataloading]
            data[key].x_ori = item[f"{key}_b_seq"]  # _seq is residue id # !!! [caution: bound seq is used here]
            data[key].x = None

            # [TODO: this is not used]
            if self.args.use_orientation_features:
                data[key].n_i_feat = item[f"{key}_u_n_i_feat"].float()
                data[key].u_i_feat = item[f"{key}_u_u_i_feat"].float()
                data[key].v_i_feat = item[f"{key}_u_v_i_feat"].float()

            # [!!!!!!!!!!! mask]
            data[key].mask = item[f"{key}_mask"]

            # bb_atoms
            data[key].bb_atoms_b = item[f"{key}_b_bb_coords"].float()
            data[key].bb_atoms_u = item[f"{key}_u_bb_coords"].float()

            # bb rotvec
            mask = item[f"{key}_mask"]

            np_bb_atoms_b = np.array(data[key].bb_atoms_b)
            np_bb_atoms_u = np.array(data[key].bb_atoms_u[mask])

            rotvecs=[]
            for i in range(np_bb_atoms_b.shape[0]):
                tran, rot = get_align_rotran(np_bb_atoms_b[i], np_bb_atoms_u[i])
                rotvecs.append(Rotation.from_matrix(rot.T).as_rotvec())
            
            torch_rotvecs = torch.from_numpy(np.array(rotvecs)).float()

            data[key].rotvecs = torch_rotvecs

            # [!!!!!!!!!!! mask]
            # ub: linear graph
            # b: linear graph with some edges deleted according to the mask
            
            #
            # ub
            num_nodes_u = data[key].pos_u.size(0)

            # Creating a linear graph
            # Connect each node to the next one
            edges = [[i, i + 1] for i in range(num_nodes_u - 1)]

            # Convert edges to a tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            data[key].edge_index_u = edge_index

            #
            # b
            num_nodes_b = data[key].pos_b.size(0)

            # Creating a linear graph
            # Connect each node to the next one
            edges = [[i, i + 1] for i in range(num_nodes_b - 1)]

            # find out where the False's in the mask begin and do not count the previous False's
            mask = item[f"{key}_mask"]

            del_edge_ind = []
            prev = True
            for i in range(len(mask)-1):
                if sum(mask[i:]) == 0:
                    break
                if prev and mask[i] == False and i != 0:
                    del_edge_ind.append(sum(mask[:i])-1)
                
                prev = mask[i]
            
            # delete the edges reversely
            for i in del_edge_ind[::-1]:
                del edges[i]

            edges = np.array(edges)

            srcs = np.concatenate([edges[:,1].copy(), edges[:,0].copy()])
            dsts = np.concatenate([edges[:,0].copy(), edges[:,1].copy()])

            edges = np.stack([srcs, dsts], axis=0)
            edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
            
            data[key].edge_index_b = edge_index

            data[key, "contact", key].edge_index = None
            # [!!!!!!!!!!! mask]

        # center receptor at origin (with bound center for both the u and b)
        center_b = data["receptor"].pos_b.mean(dim=0, keepdim=True)
        center_u = data["receptor"].pos_u.mean(dim=0, keepdim=True)
        center = data["receptor"].pos.mean(dim=0, keepdim=True)
        center_b_ori = data["receptor"].pos_b_ori.mean(dim=0, keepdim=True)

        for key in ["receptor", "ligand"]: 
            # remove bound center for the receptor for all
            # bc when doing the inference, the denoised u_l will be compared to the b_l in RMSD sense
            data[key].pos = data[key].pos - center_b
            data[key].pos_b = data[key].pos_b - center_b
            data[key].pos_u = data[key].pos_u - center_b
            data[key].pos_b_ori = data[key].pos_b_ori - center_b
            data[key].bb_atoms_b = data[key].bb_atoms_b - center_b
            data[key].bb_atoms_u = data[key].bb_atoms_u - center_b

        # [!!!!!!!!!!! mask]
        data.center = center_b  # save optimized bound center
        return data

    def compute_embeddings(self, data):
        """
            Pre-compute ESM2 embeddings.
        """
        if self.args.data_process and self.args.totalInd > 1:
            # check if we already computed embeddings
            if (not self.flag) and os.path.exists(self.esm_cache  + str(self.args.ind)) and (not self.args.recache):
                with open(self.esm_cache  + str(self.args.ind), "rb") as f:
                    path_to_rep = pickle.load(f)
                # ???
                print("here")
                self._save_esm_rep(data, path_to_rep)
                printt("Loaded cached ESM embeddings")
                return data
        else:
            # check if we already computed embeddings
            if (not self.flag) and os.path.exists(self.esm_cache) and (not self.args.recache):
                with open(self.esm_cache, "rb") as f:
                    path_to_rep = pickle.load(f)
                # ???
                print("here")
                self._save_esm_rep(data, path_to_rep)
                printt("Loaded cached ESM embeddings")
                return data

        printt("Computing ESM embeddings")
        # load pretrained model
        esm_model, alphabet = torch.hub.load(
                "facebookresearch/esm:main",
                "esm2_t33_650M_UR50D")
        self.esm_model = esm_model.cuda().eval()
        tokenizer = alphabet.get_batch_converter()
        # convert to 3 letter codes
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {k.upper():v for k,v in protein_letters_3to1.items()})
        # fix ordering
        all_pdbs = sorted(data)
        all_graphs = [data[pdb]["graph"] for pdb in all_pdbs]
        rec_seqs = [g["receptor"].x_ori for g in all_graphs]
        lig_seqs = [g["ligand"].x_ori for g in all_graphs]
        rec_seqs = ["".join(aa_code[s] for s in seq) for seq in rec_seqs]
        lig_seqs = ["".join(aa_code[s] for s in seq) for seq in lig_seqs]
        # batchify sequences
        rec_batches = self._esm_batchify(rec_seqs, tokenizer)
        lig_batches = self._esm_batchify(lig_seqs, tokenizer)
        with torch.no_grad():
            pad_idx = alphabet.padding_idx # tipically a scalar value
            rec_reps = self._run_esm(rec_batches, pad_idx)
            lig_reps = self._run_esm(lig_batches, pad_idx)

        # dump to cache
        path_to_rep = {}
        for idx, pdb in enumerate(all_pdbs):
            # cat ESM embedding and one-hot representation
            rec_graph_x = torch.cat([rec_reps[idx][0],
                rec_reps[idx][1]], dim=1)
            lig_graph_x = torch.cat([lig_reps[idx][0],
                lig_reps[idx][1]], dim=1)
            path_to_rep[pdb] = rec_graph_x, lig_graph_x
        
        # ???
        if self.args.data_process and self.args.totalInd > 1:
            with open(self.esm_cache  + str(self.args.ind), "wb+") as f:
                pickle.dump(path_to_rep, f)
        else:
            with open(self.esm_cache, "wb+") as f:
                pickle.dump(path_to_rep, f)
        



        # overwrite graph.x_ori for each element in batch
        self._save_esm_rep(data, path_to_rep)

        return data

    def _esm_batchify(self, seqs, tokenizer):
        batch_size = find_largest_diviser_smaller_than(len(seqs), 6)
        #self.args.batch_size if len(seqs)%self.args.batch_size == 0 else find_smallest_diviser(len(seqs))

        iterator = range(0, len(seqs), batch_size)
        # group up sequences
        batches = [seqs[i:i + batch_size] for i in iterator]
        batches = [[("", seq) for seq in batch] for batch in batches]
        # tokenize
        batch_tokens = [tokenizer(batch)[2] for batch in batches]
        return batch_tokens

    def _run_esm(self, batches, padding_idx):
        """
            Wrapper around ESM specifics
            @param (list)  batch
            @return (list)  same order as batch
        """
        # run ESM model
        all_reps = []
        for batch in tqdm(batches, desc="ESM", ncols=50):
            reps = self.esm_model(batch.cuda(), repr_layers=[33])
            #reps = reps["representations"][33].cpu().squeeze()[:,1:]
            # remove squeeze() to allow for running on a single pair.
            reps = reps["representations"][33].cpu()[:,1:]
            all_reps.append(reps)
        # crop to length
        # exclude <cls> <sep>
        cropped = []
        for i, batch in enumerate(batches):
            batch_lens = (batch != padding_idx).sum(1) - 2
            for j, length in enumerate(batch_lens):
                rep_crop = all_reps[i][j,:length]
                token_crop = batch[j,1:length+1,None]
                cropped.append((rep_crop, token_crop))
        return cropped

    def _save_esm_rep(self, data, path_to_rep):
        """
            Assign new ESM representation to graph.x_ori
        """

        # new_data = {}

        ct = 0
        cc = 0

        for pdb, (rec_rep, lig_rep) in tqdm(path_to_rep.items(), desc="_save_esm_rep"):
            # ???
            print(pdb)
            
            if pdb not in data:
                ct += 1
                continue

            rec_graph = data[pdb]["graph"]["receptor"]
            lig_graph = data[pdb]["graph"]["ligand"]

            if len(rec_graph.pos_b) != len(rec_rep) or len(lig_graph.pos_b) != len(lig_rep):
                data.pop(pdb, None)
                print(f'skip {pdb} bc lengths mismatch')
                cc += 1
                continue
            
            rec_graph.x_ori = rec_rep
            lig_graph.x_ori = lig_rep
            assert len(rec_graph.pos_b) == len(rec_graph.x_ori)
            # print(f'len(rec_graph.pos): {len(rec_graph.pos)}')
            # print(f'len(rec_graph.x_ori): {len(rec_graph.x_ori)}')
            # print('***')
            # print(f'len(lig_graph.pos): {len(lig_graph.pos)}')
            # print(f'len(lig_graph.x_ori): {len(lig_graph.x_ori)}')
            # print('-----')
            assert len(lig_graph.pos_b) == len(lig_graph.x_ori)

            
        
        print(f'total of {ct} complexes skipped bc of res number less than {self.args.filter_len_r_or_l}')
        print(f'total of {cc} complexes skipped bc of esm rep length mismatch with graph res number')
        
        return data

    def split_data(self, raw_data, args):
        '''
        Returns:
            1. data: ({"train": [data], "val": data, "test": data})
        '''
        # separate out train/test
        data = {
            "train": {},
            "val":   {},
            "test":  {}
        }
        ## if test split is pre-specified, split into train/test
        # otherwise, allocate all data to train for cross-validation
        for k, item in raw_data.items():
            if "split" in item:
                data[item["split"]][k] = item
            else:
                data["train"] = raw_data
                break
        # split train into separate folds
        # sometimes we only want to load val data. Then train data is empty
        if len(data["train"]) > 0:
            data["train"] = split_into_folds(data["train"], args) # Return: 1. folds: ([dict]) a list of dictionaries containing the original data entries {pdbs: data.values()}
        
        print("finished split_data")

        return data

    def crossval_split(self, fold_num):
        """
            number of folds-way cross validation
            @return  dict: split -> [pdb_ids]
        """
        splits = { "train": [] }
        # # split into train/val/test
        # folds = [list(sorted(f)) for f in self.splits["train"]]
        # val_data = list(sorted(self.splits["val"]))
        # test_data = list(sorted(self.splits["test"]))

        # [Modified]
        # self.splits_to_pdbList
        # split into train/val/test
        folds = [list(sorted(f)) for f in self.splits_to_pdbList["train"]]
        val_data = list(sorted(self.splits_to_pdbList["val"]))
        test_data = list(sorted(self.splits_to_pdbList["test"]))

        # if val split is pre-specified, do not take fold
        if len(val_data) == 0:
            # pass
            val_fold = fold_num
            splits["val"] = folds[val_fold]
        else:
            val_fold = -1  # all remaining folds go to train
            splits["val"] = val_data
        # if test split is pre-specified, do not take fold
        if len(test_data) > 0:
            test_fold = val_fold  # must specify to allocate train folds
            splits["test"] = test_data
        # otherwise both val/test labels depend on fold_num
        else:
            test_fold = (fold_num+1) % self.args.num_folds
            splits["test"] = folds[test_fold]
        # add remaining to train
        for idx in range(self.args.num_folds):
            if idx in [val_fold, test_fold]:
                continue
            splits["train"].extend(folds[idx])
        return splits


class DIPSLoader(Loader):
    def __init__(self, args, split=None, batch=None, verbose=True):
        super(DIPSLoader, self).__init__(args)

        # overall mask file
        flder_csv = os.path.dirname(self.data_file)
        # pth_mask = os.path.join(flder_csv, "mask.pkl")

        self.mask = pickle.load(open(os.path.join(flder_csv, "mask.pkl"), "rb"))

        self.data = None # {pdb_id: {k:v}}
        self.data_params = None # {k_params: v_params}
        self.dict_pdb_to_pklPath = None # {pdb_id: pkl_path}
        self.splits = None # {"train": [data], "val": data, "test": data}
        self.splits_to_pdbList = None # {"train": split: [[pdb_id]], "val": [pdb_id], "test": [pdb_id]}
        self.dict_pdb_to_totalLen = None # {pdb_id: len_r + len_l}

        # check if self.args.filter_len_r_or_l is 0 or not
        # if 0, set it to math.ceil((args.n_NM + 6 ) / 3)
        if self.args.filter_len_r_or_l == 0:
            self.args.filter_len_r_or_l = math.ceil((args.n_NM + 6 ) / 3)

        # if no files exist, compute and save them (all with subindex (ind))
        # if exist, load for self.data_params

        # target: data_params, sep_pkl, dict_pdb_to_pklPath.pkl, splits_to_pdbList.pkl, dict_pdb_to_totalLen.pkl
        # (combined afterwards)
 

        directory_name = self.data_file.replace(".csv", "_splits_to_pdbList") 
        folder_path = os.path.join(self.root, directory_name)

        # if (not os.path.exists(os.path.join(folder_path, "splits_to_pdbList.pkl"))) and (not self.args.recache) and (not self.args.data_process):
        #     try:
        #         construct_dict = {'train':[], 'val':[], 'test':[]}
        #         # read in the csv file with path self.data_file
        #         with open(self.data_file, 'r') as f:
        #             reader = csv.reader(f)
        #             # read the first line as headers
        #             headers = next(reader)

        #             sep_pkl_directory_name = self.data_file.replace(".csv", "_sep_pkl") 
        #             sep_pkl_folder_path = os.path.join(self.root, sep_pkl_directory_name)



        #             if headers[0] != "path" and os.path.exists(os.path.join(sep_pkl_folder_path, headers[0]+'.pkl')):
        #                 pdb = headers[0]
        #                 split = headers[1]
        #                 construct_dict[split].append(pdb)
                    
        #             for row in reader:
        #                 # read in the row
        #                 if os.path.exists(os.path.join(sep_pkl_folder_path, row[0]+'.pkl')):
        #                     pdb = row[0]
        #                     split = row[1]
        #                     construct_dict[split].append(pdb)
        #         pickle.dump(construct_dict, open(os.path.join(folder_path, "splits_to_pdbList.pkl"), "wb"))

        #     except:
        #         pass




                

        # !!!!
        # force to recompute
        self.flag = False
        # self.flag = True ########

        # print('self.args.recache', self.args.recache)
        # print('flag2', not os.path.exists(os.path.join(folder_path, "splits_to_pdbList.pkl")))

        if self.flag or (not os.path.exists(os.path.join(folder_path, "splits_to_pdbList.pkl"))) or self.args.recache:
            print("full processing !!!")
                      
            # ???
            # self.flag = False ########
            # load and process files

            # ???
            data = self.read_files() 

            # ???
            data = self.assign_receptor(data)

            # ???
            # self.flag = True

            # ???
            # data, data_params = self.process_data(self.data, args)
            # ??? [ori]
            data, data_params = self.process_data(data, args)

            # ???
            # self.flag = True #########
            # ???
            # print(aaa)
            data, data_params = self.process_embed(data, data_params)

            # ???
            # self.flag = True

            #### pre-compute ESM embeddings if needed
            self.data = data
            self.data_params = data_params

            directory_name = self.data_file.replace(".csv", "_processed_data_param") 
            folder_path = os.path.join(self.root, directory_name)

            os.makedirs(folder_path, exist_ok=True)

            if self.args.data_process and self.args.totalInd > 1:
                if self.flag or (not os.path.exists(os.path.join(folder_path, f"data.pkl{self.args.ind}"))):
                    pickle.dump(data, open(os.path.join(folder_path, f"data.pkl{self.args.ind}"), "wb"))
                    pickle.dump(data_params, open(os.path.join(folder_path, f"data_params.pkl{self.args.ind}"), "wb"))
            else:
                if self.flag or (not os.path.exists(os.path.join(folder_path, "data.pkl"))):
                    pickle.dump(data, open(os.path.join(folder_path, "data.pkl"), "wb"))
                    pickle.dump(data_params, open(os.path.join(folder_path, "data_params.pkl"), "wb"))
            printt(len(self.data), "entries loaded")



            # [Modify: lazy loader]
            self.dict_pdb_to_pklPath = self.constr_dict_pdb_to_pklPath()

            # split into folds
            self.splits = self.split_data(data, args)

            directory_name = self.data_file.replace(".csv", "_splits") 
            folder_path = os.path.join(self.root, directory_name)

            os.makedirs(folder_path, exist_ok=True)

            if self.args.data_process and self.args.totalInd > 1:
                if self.flag or (not os.path.exists(os.path.join(folder_path, f"splits.pkl{self.args.ind}"))):
                    pickle.dump(self.splits, open(os.path.join(folder_path, f"splits.pkl{self.args.ind}"), "wb"))
                    pickle.dump(self.splits["test"], open(os.path.join(folder_path, f"splits_test.pkl{self.args.ind}"), "wb"))
                    pickle.dump(self.splits["val"], open(os.path.join(folder_path, f"splits_val.pkl{self.args.ind}"), "wb"))
            else:
                if self.flag or (not os.path.exists(os.path.join(folder_path, "splits.pkl"))):
                    pickle.dump(self.splits, open(os.path.join(folder_path, "splits.pkl"), "wb"))
                    pickle.dump(self.splits["test"], open(os.path.join(folder_path, "splits_test.pkl"), "wb"))
                    pickle.dump(self.splits["val"], open(os.path.join(folder_path, "splits_val.pkl"), "wb"))

            # construct self.splits_to_pdbList
            self.splits_to_pdbList = self.splits_to_splits_to_pdbList()

            print("finished constructing self.splits_to_pdbList")

            # construct self.dict_pdb_to_totalLen for dataloader
            self.dict_pdb_to_totalLen = self.constr_dict_pdb_to_totalLen()
        
        else:
            printt("Reading all the cached files!!!!")

            directory_name = self.data_file.replace(".csv", "_processed_data_param") 
            folder_path = os.path.join(self.root, directory_name)

            self.data_params = pickle.load(open(os.path.join(folder_path, "data_params.pkl"), "rb"))

            
            # [Modify: lazy loader]
            self.dict_pdb_to_pklPath = self.constr_dict_pdb_to_pklPath()

            # construct self.splits_to_pdbList
            self.splits_to_pdbList = self.splits_to_splits_to_pdbList()

            # construct self.dict_pdb_to_totalLen for dataloader
            self.dict_pdb_to_totalLen = self.constr_dict_pdb_to_totalLen()

        
        # print(aaaaa)


        '''
        # load and process files
        data = self.read_files()
        data = self.assign_receptor(data)
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)

        #### pre-compute ESM embeddings if needed
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")

        # [Modify: lazy loader]
        self.dict_pdb_to_pklPath = self.constr_dict_pdb_to_pklPath()

        # split into folds
        self.splits = self.split_data(data, args)
        '''

    def constr_dict_pdb_to_totalLen(self):

        #
        # load/compute dict_pdb_to_totalLen
        # check dict_pdb_to_totalLen if cached [TODO: do this in the data preprocessing]
        directory_name = self.data_file.replace(".csv", "dict_pdb_to_totalLen") 
        folder_path = os.path.join(self.root, directory_name)

        if self.args.data_process and self.args.totalInd > 1:
            if self.flag or (not os.path.exists(os.path.join(folder_path, f"dict_pdb_to_totalLen.pkl{self.args.ind}"))) or self.args.recache:  
                os.makedirs(folder_path, exist_ok=True)
                dict_pdb_to_totalLen = {} # {"ke/1kee.pdb1_6.dill": some number repreing the total complex len}
                
                data_to_load = load_csv(self.data_file) ## some data are left out !!!
                for line in tqdm(data_to_load, desc="calc com len"):

                    pdb_id = line["path"]
                    if pdb_id not in self.dict_pdb_to_pklPath:
                        continue
                    pth = self.dict_pdb_to_pklPath[pdb_id]
                    

                    each_data = pickle.load(open(pth, "rb"))

                    r_len = each_data["graph"]["receptor"].pos_b.shape[0]
                    l_len = each_data["graph"]["ligand"].pos_b.shape[0]

                    dict_pdb_to_totalLen[pdb_id] = r_len + l_len

                    # print("pdb_id", pdb_id, "r_len", r_len, "l_len", l_len)

                    # print(aaaaa)
                
                pickle.dump(dict_pdb_to_totalLen, open(os.path.join(folder_path, f"dict_pdb_to_totalLen.pkl{self.args.ind}"), "wb"))
            
            else:
                dict_pdb_to_totalLen = pickle.load(open(os.path.join(folder_path, f"dict_pdb_to_totalLen.pkl{self.args.ind}"), "rb"))

        else:
            if self.flag or (not os.path.exists(os.path.join(folder_path, "dict_pdb_to_totalLen.pkl"))) or self.args.recache:
                os.makedirs(folder_path, exist_ok=True)
                dict_pdb_to_totalLen = {} # {"ke/1kee.pdb1_6.dill": some number repreing the total complex len}
                
                data_to_load = load_csv(self.data_file)
                for line in tqdm(data_to_load, desc="calc com len"):

                    pdb_id = line["path"]
                    if pdb_id not in self.dict_pdb_to_pklPath:
                        print(f"pdb_id {pdb_id} are filtered out")
                        continue

                    # #####
                    # print(pdb_id)

                    pth = self.dict_pdb_to_pklPath[pdb_id]
                    

                    each_data = pickle.load(open(pth, "rb"))

                    r_len = each_data["graph"]["receptor"].pos_b.shape[0]
                    l_len = each_data["graph"]["ligand"].pos_b.shape[0]

                    dict_pdb_to_totalLen[pdb_id] = r_len + l_len

                    # print("pdb_id", pdb_id, "r_len", r_len, "l_len", l_len)

                    # print(aaaaa)
                
                pickle.dump(dict_pdb_to_totalLen, open(os.path.join(folder_path, "dict_pdb_to_totalLen.pkl"), "wb"))
            
            else:
                dict_pdb_to_totalLen = pickle.load(open(os.path.join(folder_path, "dict_pdb_to_totalLen.pkl"), "rb"))

        return dict_pdb_to_totalLen

    def get_splits(self):
        if self.splits:
            return self.splits
        else:
            directory_name = self.data_file.replace(".csv", "_splits") 
            folder_path = os.path.join(self.root, directory_name)

            return pickle.load(open(os.path.join(folder_path, "splits.pkl"), "rb"))


        
    def splits_to_splits_to_pdbList(self):
        '''compute splits_to_pdbList (dict): from {split: data OR [data]} to {split: [pdb_id] OR [[pdb_id]]}
        '''

        directory_name = self.data_file.replace(".csv", "_splits_to_pdbList") 
        folder_path = os.path.join(self.root, directory_name)

        if self.args.data_process and self.args.totalInd > 1:
            if self.flag or (not os.path.exists(os.path.join(folder_path, f"splits_to_pdbList.pkl{self.args.ind}"))) or self.args.recache:
                os.makedirs(folder_path, exist_ok=True)
                splits_to_pdbList = {
                "train": {},
                "val":   {},
                "test":  {}
                }
                folds = [list(sorted(f)) for f in self.splits["train"]]
                val_data = list(sorted(self.splits["val"]))
                test_data = list(sorted(self.splits["test"]))

                splits_to_pdbList["train"] = folds
                splits_to_pdbList["val"] = val_data
                splits_to_pdbList["test"] = test_data

                pickle.dump(splits_to_pdbList, open(os.path.join(folder_path, f"splits_to_pdbList.pkl{self.args.ind}"), "wb"))
                return splits_to_pdbList
            else:
                return pickle.load(open(os.path.join(folder_path, f"splits_to_pdbList.pkl{self.args.ind}"), "rb"))
        else:
            if self.flag or (not os.path.exists(os.path.join(folder_path, "splits_to_pdbList.pkl"))) or self.args.recache:
                os.makedirs(folder_path, exist_ok=True)
                splits_to_pdbList = {
                "train": {},
                "val":   {},
                "test":  {}
                }
                folds = [list(sorted(f)) for f in self.splits["train"]]
                val_data = list(sorted(self.splits["val"]))
                test_data = list(sorted(self.splits["test"]))

                splits_to_pdbList["train"] = folds
                splits_to_pdbList["val"] = val_data
                splits_to_pdbList["test"] = test_data

                pickle.dump(splits_to_pdbList, open(os.path.join(folder_path, "splits_to_pdbList.pkl"), "wb"))
                return splits_to_pdbList
            else:
                return pickle.load(open(os.path.join(folder_path, "splits_to_pdbList.pkl"), "rb"))






        
    
    def constr_dict_pdb_to_pklPath(self):
        
        directory_name = self.data_file.replace(".csv", "_sep_pkl") 
        folder_path = os.path.join(self.root, directory_name)

        dict_pdb_to_pklPath = {}

        if self.args.data_process and self.args.totalInd > 1:
            # Check if the directory already exists, and create it if not
            if self.flag or (not os.path.exists(os.path.join(folder_path, f"dict_pdb_to_pklPath.pkl{self.args.ind}"))) or self.args.recache:
                os.makedirs(folder_path, exist_ok=True)
                for k,v in tqdm(self.data.items(), desc="Processing separate pickle files"):
                    # k is the pdb_name, v is the value dictionary

                    pdb_pkl_path = os.path.join(folder_path, k + ".pkl")
                    folder, _ = os.path.split(pdb_pkl_path)
                    os.makedirs(folder, exist_ok=True)
                    
                    pickle.dump(v, open(pdb_pkl_path, "wb"))
                    dict_pdb_to_pklPath[k] = pdb_pkl_path
                
                pickle.dump(dict_pdb_to_pklPath, open(os.path.join(folder_path, f"dict_pdb_to_pklPath.pkl{self.args.ind}"), "wb"))
                return dict_pdb_to_pklPath
            else:
                return pickle.load(open(os.path.join(folder_path, f"dict_pdb_to_pklPath.pkl{self.args.ind}"), "rb"))
         
        else:
            # Check if the directory already exists, and create it if not
            if self.flag or (not os.path.exists(os.path.join(folder_path, "dict_pdb_to_pklPath.pkl"))) or self.args.recache:
                os.makedirs(folder_path, exist_ok=True)
                for k,v in tqdm(self.data.items(), desc="Processing separate pickle files"):
                    # k is the pdb_name, v is the value dictionary

                    pdb_pkl_path = os.path.join(folder_path, k + ".pkl")
                    folder, _ = os.path.split(pdb_pkl_path)
                    os.makedirs(folder, exist_ok=True)
                    
                    pickle.dump(v, open(pdb_pkl_path, "wb"))
                    dict_pdb_to_pklPath[k] = pdb_pkl_path
                
                pickle.dump(dict_pdb_to_pklPath, open(os.path.join(folder_path, "dict_pdb_to_pklPath.pkl"), "wb"))
                return dict_pdb_to_pklPath
            else:
                return pickle.load(open(os.path.join(folder_path, "dict_pdb_to_pklPath.pkl"), "rb"))
         

    # [nm]
    def read_files(self):

        '''data
        should be in the format of 
        {path: 
            {
            "pdb_id": pdb/path, # eg. 'ke/1kee.pdb1_6.dill'
            "receptor_b": p1_b,
            "ligand_b": p2_b,
            "receptor_u": p1_u,
            "ligand_u": p2_u,
            'path': pdb/path, 
            'split': 
            }
        }
        '''
        data = {}
        # check if loaded previously
        # [Modified]
        if self.args.data_process and self.args.totalInd > 1:
            boolean_value = (not self.flag) and (not self.args.recache and os.path.exists(self.data_cache + str(self.args.ind)))
            load_path = self.data_cache + str(self.args.ind)
        else:
            boolean_value = (not self.flag) and (not self.args.recache and os.path.exists(self.data_cache))
            load_path = self.data_cache

        if boolean_value:
            with open(load_path, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        
        #
        # path_to_data
        # self.data_to_load = [{'path': , 'split': }]

        # ???
        length = len(self.data_to_load)
        step = length // self.args.totalInd + 1
        # ???
        if self.args.data_process and self.args.totalInd > 1:
            tqdm_obj = tqdm(self.data_to_load[(self.args.ind)*step: (self.args.ind + 1)*step], desc="data loading", ncols=50)
        else:
            tqdm_obj = tqdm(self.data_to_load, desc="data loading", ncols=50)
        # for line in tqdm(self.data_to_load[(self.args.ind)*step: (self.args.ind + 1)*step],
        # for line in tqdm(self.data_to_load,
        #                  desc="data loading", ncols=50):
        for line in tqdm_obj:
            path = line["path"] # 'ke/1kee.pdb1_6.dill'
            item = path_to_data.get(path)

            if not self.args.model_type == 'confidence':
                if (not os.path.exists(os.path.join(self.root, path)[:-5] + '.bound.dill')) or (not os.path.exists(os.path.join(self.root, path))):
                    print(f"skip {path} because either bound or unbound dill file is missing")
                    continue
            else:
                if (not os.path.exists(os.path.join(self.root, path[:7]) + '.bound.dill')) or (not os.path.exists(os.path.join(self.root, path))):
                    print(f"skip {path} because either bound or unbound dill file is missing")
                    continue

            
            if item is None:

                # parse_dill_nm(self, fp_b, fp_u, pdb_id)
                # fp_b: 'datasets/DIPS/pairs_pruned/ke/1kee.pdb1_6.bound.dill'
                # fp_u: 'datasets/DIPS/pairs_pruned/ke/1kee.pdb1_6.dill'
                # pdb_id: 'ke/1kee.pdb1_6.dill'
                if not self.args.model_type == 'confidence':
                    item = self.parse_dill_nm(os.path.join(self.root, path)[:-5] + '.bound.dill', os.path.join(self.root, path), path)
                else:
                    item = self.parse_dill_nm(os.path.join(self.root, path[:7]) + '.bound.dill', os.path.join(self.root, path), path)
                

                #[!!!]
                # print("rec[0]", item["receptor_b"][0])
                # print(aaaaa)


                path_to_data[path] = item
            
            item.update(line)  # add meta-data

            # item = {
            # "pdb_id": pdb,
            # "receptor_b": p1_b,
            # "ligand_b": p2_b,
            # "receptor_u": p1_u,
            # "ligand_u": p2_u,
            # 'path': , 
            # 'split': 
            # }

            data[path] = item

            if self.args.debug:
                if len(data) >= 2:
                    break


        # write to cache
        # ???
        if self.args.data_process and self.args.totalInd > 1:
            boolean_value = (self.flag) or self.args.recache or not os.path.exists(self.data_cache + str(self.args.ind))
            save_path = self.data_cache + str(self.args.ind)
        else:
            boolean_value = (self.flag) or self.args.recache or not os.path.exists(self.data_cache)
            save_path = self.data_cache
        
        # if (self.flag) or self.args.recache or not os.path.exists(self.data_cache + str(self.args.ind)):
        # if (self.flag) or self.args.recache or not os.path.exists(self.data_cache):
        if boolean_value:
            # ???
            # with open(self.data_cache  + str(self.args.ind), "wb+") as f:
            # with open(self.data_cache, "wb+") as f:
            with open(save_path, "wb+") as f:
                pickle.dump(path_to_data, f)

        return data

    # [nm]
    def assign_receptor(self, data):
        """
            For docking, we assigned smaller protein as ligand
            for this dataset (since no canonical receptor/ligand
            assignments)
        """
        # new_data = {}
        # ct = 0
        for key, item in tqdm(data.items(), desc='in assign_receptor'):

            rec = item["receptor_b"]
            lig = item["ligand_b"]

            # [!!!]
            # print("rec[0]", rec[0])
            # print(aaaa)

            # if len(rec[0]) < self.args.filter_len_r_or_l or len(lig[0]) < self.args.filter_len_r_or_l:
            #     print(f'skip {key} because r or l is too small in the complex.')
            #     ct += 1
            #     continue

            if len(rec[0]) < len(lig[0]):
                item["receptor_b"] = lig
                item["ligand_b"] = rec

                rec = item["receptor_u"]
                lig = item["ligand_u"]

                item["receptor_u"] = lig
                item["ligand_u"] = rec

                if not self.args.model_type == 'confidence':
                    # mask 
                    self.mask[key[3:-5]] = {'l': self.mask[key[3:-5]]['r'], 'r': self.mask[key[3:-5]]['l']} 
                # else:
                #     # mask  
                #     self.mask[key[3:7]] = {'l': self.mask[key[3:7]]['r'], 'r': self.mask[key[3:7]]['l']}
                            

            if self.args.model_type == 'confidence':
                rec_b_len = sum(1 for i in item["receptor_b"][1] if i[0] == 'CA')
                lig_b_len = sum(1 for i in item["ligand_b"][1] if i[0] == 'CA')
                rec_u_len = sum(1 for i in item["receptor_u"][1] if i[0] == 'CA')
                lig_u_len = sum(1 for i in item["ligand_u"][1] if i[0] == 'CA')

                if rec_b_len != rec_u_len or lig_b_len != lig_u_len:
                    print(f"{key} with rec_b_len != rec_u_len or lig_b_len != lig_u_len")
                    
                assert rec_b_len == rec_u_len
                assert lig_b_len == lig_u_len

                self.mask[key[3:7]] = {'l': [True] * lig_b_len, 'r': [True] * rec_b_len}

                # # mask 
                # if len(self.mask[key[3:7]]['r']) < len(self.mask[key[3:7]]['l']):
                #     self.mask[key[3:7]] = {'l': self.mask[key[3:7]]['r'], 'r': self.mask[key[3:7]]['l']}
                            
            # new_data[key] = item
        
        # print(f'skip {ct} complexes because either r or l is too small in those complexes.')

        return data

    def parse_dill(self, fp, pdb_id):
        with open(fp, "rb") as f:
            data = dill.load(f)
        p1, p2 = data[1], data[2]
        p1, p2 = self.parse_df(p1), self.parse_df(p2)
        return self._to_dict(pdb_id, p1, p2)
    
    # [nm]
    def parse_dill_nm(self, fp_b, fp_u, pdb_id):
        '''
            Return 
            item = {
            "pdb_id": pdb,
            "receptor_b": p1_b,
            "ligand_b": p2_b,
            "receptor_u": p1_u,
            "ligand_u": p2_u,
        }
        '''
        with open(fp_b, "rb") as f:
            data_b = dill.load(f)
        with open(fp_u, "rb") as f:
            data_u = dill.load(f)
        p1_b, p2_b = data_b[1], data_b[2]
        p1_u, p2_u = data_u[1], data_u[2]
        p1_b, p2_b = self.parse_df(p1_b), self.parse_df(p2_b)
        p1_u, p2_u = self.parse_df(p1_u), self.parse_df(p2_u)

        if self.args.model_type == 'confidence':
            # check the correspondence between bound and unbound in case r/l swapped
            len_p1_b = sum(1 for i in p1_b[1] if i[0] == 'CA')
            len_p2_b = sum(1 for i in p2_b[1] if i[0] == 'CA')
            len_p1_u = sum(1 for i in p1_u[1] if i[0] == 'CA')
            len_p2_u = sum(1 for i in p2_u[1] if i[0] == 'CA')

            if len_p1_b != len_p1_u or len_p2_b != len_p2_u:
                # swap r/l for unbound
                p1_u, p2_u = p2_u, p1_u
                print(f"Swapped r/l for {pdb_id} bc len_p1_b != len_p1_u or len_p2_b != len_p2_u")

            elif (len_p1_u == len_p2_u) and (p1_b[0][0] != p1_u[0][0]):
                # swap r/l for unbound
                p1_u, p2_u = p2_u, p1_u
                print(f"Swapped r/l for {pdb_id} bc p1_b[0][0] != p1_u[0][0]")


            
            
                
        return self._to_dict_nm(pdb_id, p1_b, p2_b, p1_u, p2_u)

    def parse_df(self, df):
        """
            Parse PDB DataFrame
        """
        # extract dataframe values
        all_res = df["resname"]
        all_pos = torch.tensor([df["x"], df["y"], df["z"]]).t()
        all_atom = list(zip(df["atom_name"], df["element"]))
        visualization_values = {"chain": df["chain"],
                                "resname": all_res,
                                "residue": df["residue"],
                                "atom_name": df["atom_name"],
                                "element": df["element"]}
        # convert to seq, pos
        return all_res, all_atom, all_pos, visualization_values


class DB5Loader(Loader):
    """
        Docking benchmark 5.5
    """
    def __init__(self, args):
        super(DB5Loader, self).__init__(args)
        # load and process files
        self.root = os.path.join(self.root, "structures")
        # cache file dependent on use_unbound
        if args.use_unbound:
            printt('Using Unbound structures')
            self.data_cache = self.data_cache.replace(".pkl", "_u.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_u.pkl")
        else:
            printt('Using Bound structures')
            self.data_cache = self.data_cache.replace(".pkl", "_b.pkl")
            self.esm_cache = self.esm_cache.replace(".pkl", "_b.pkl")
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if not self.args.recache and os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}
        for line in tqdm(self.data_to_load,
                         desc="data loading", ncols=50):
            pdb = line["path"]
            item = path_to_data.get(pdb)
            if item is None:
                item = self.parse_path(pdb, os.path.join(self.root, pdb))
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
            if self.args.debug:
                if len(data) >= 2:
                    break
        # write to cache
        if self.args.recache or not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data

    def parse_path(self, pdb, path):
        if self.args.use_unbound:
            fp_rec = f"{path}_r_u.pdb"
            fp_lig = f"{path}_l_u.pdb"
        else:
            fp_rec = f"{path}_r_b.pdb"
            fp_lig = f"{path}_l_b.pdb"
        p1, p2 = self.parse_pdb(fp_rec), self.parse_pdb(fp_lig)
        return self._to_dict(pdb, p1, p2)


class SabDabLoader(Loader):
    """
        Structure Antibody Database
        Downloaded May 2, 2022.
    """
    def __init__(self, args):
        super(SabDabLoader, self).__init__(args)
        # standard workflow
        data = self.read_files()
        data, data_params = self.process_data(data, args)
        data, data_params = self.process_embed(data, data_params)
        self.data = data
        self.data_params = data_params
        printt(len(self.data), "entries loaded")
        # split into folds
        self.splits = self.split_data(data, args)

    def read_files(self):
        data = {}
        # check if loaded previously
        if os.path.exists(self.data_cache):
            with open(self.data_cache, "rb") as f:
                path_to_data = pickle.load(f)
        else:
            path_to_data = {}

        for line in tqdm(self.data_to_load,
                         desc="data loading", ncols=50):
            pdb = line["pdb"]
            item = path_to_data.get(pdb)
            if item is None:
                # parse sabdab entry
                pdb = line["pdb"]
                rec_chains = [
                    c.strip() for c in line["antigen_chain"].split("|")]
                lig_chains = set([c.upper() for c in [
                    line["Hchain"], line["Lchain"]] if c is not None])
                # parse pdb object
                path = os.path.join(self.root, f"{pdb}.pdb")
                rec = self.parse_pdb(path, line["model"], rec_chains)
                lig = self.parse_pdb(path, line["model"], lig_chains)
                # TODO SabDab pdb_id needs to be de-duped
                item = self._to_dict(pdb, rec, lig)
                path_to_data[pdb] = item
            item.update(line)  # add meta-data
            data[pdb] = item
        # write to cache
        if not os.path.exists(self.data_cache):
            with open(self.data_cache, "wb+") as f:
                pickle.dump(path_to_data, f)
        return data


# ------ DATA PROCESSING ------

def tokenize(data, key, tokenizer=None):
    """
        Tokenize every item[key] in data.
        Modifies item[key] and copies original value to item[key_raw].
        @param (list) data
        @param (str) key  item[key] is iterable
    """
    if len(data) == 0:  # sometimes no val split, etc.
        return
    # if tokenizer is not provided, create index
    all_values = [item[key] for item in data]
    all_values = set(itertools.chain(*all_values))
    if tokenizer is None:
        tokenizer = {}  # never used
    if type(tokenizer) is dict:
        for item in sorted(all_values):
            if item not in tokenizer:
                tokenizer[item] = len(tokenizer) + 1  # 1-index
        f_token = lambda seq: [tokenizer[x] for x in seq]
    else:
        aa_code = defaultdict(lambda: "<unk>")
        aa_code.update(
            {k.upper():v for k,v in protein_letters_3to1.items()})
        def f_token(seq):
            seq = "".join([aa_code[s] for s in seq])
            seq = tokenizer([("", seq)])[2][0]
            return seq
    # tokenize items and modify data in place
    raw_key = f"{key}_raw"
    for item in data:
        raw_item = item[key]
        item[raw_key] = raw_item  # save old
        # tokenize and convert to tensor if applicable
        item[key] = f_token(raw_item)
        if not torch.is_tensor(item[key]):
            item[key] = torch.tensor(item[key])
    return tokenizer


def split_into_folds(data, args):
    """
        Split into train/val/test folds
        @param (list) data
    """
    # split by cluster
    for item in data.values():
        by_cluster = ("cluster" in item)
        break
    if by_cluster:
        clst_to_pdb = defaultdict(list)
        for k in data:
            clst_to_pdb[data[k]["cluster"]].append(k)
    else:
        clst_to_pdb = {k:[k] for k in data}
    keys = list(sorted(clst_to_pdb))
    random.shuffle(keys)
    # split into folds
    cur_clst = 0
    folds = [{} for _ in range(args.num_folds)]
    fold_ratio = 1. / args.num_folds
    max_fold_size = int(len(data) * fold_ratio)
    for fold_num in range(args.num_folds):
        fold = []
        while (len(fold) < max_fold_size) and (cur_clst < len(clst_to_pdb)):
            fold.extend(clst_to_pdb[keys[cur_clst]])
            cur_clst += 1
        for k in fold:
            folds[fold_num][k] = data[k]
    return folds

# ------ DATA COLLATION -------

def get_mask(lens):
    """ torch.MHA style mask (False = attend, True = mask) """
    mask = torch.arange(max(lens))[None, :] >= lens[:, None]
    return mask

def find_smallest_diviser(n):
    """
    Returns smallest diviser > 1
    """
    i = 2
    while n%i != 0:
        i += 1
    return i

def find_largest_diviser_smaller_than(n, upper):
    """
    Returns largest diviser of n smaller than upper
    """
    i = upper
    while n%i !=0:
        i = i-1
    return i

def get_align_rotran(coords,reference_coords):
    '''coords@rot + tran = reference_coords
        Input: coords/reference_coords: (n,ca,c) shape: (3,3)
        Return: tran: shape:(1,3), rot: shape:(3,3)
    '''
    # center on centroid
    av1 = coords[[1],:]
    av2 = reference_coords[[1],:]
    coords = coords - av1
    reference_coords = reference_coords - av2
    # correlation matrix
    a = (coords.T)@reference_coords
    u, d, vt = np.linalg.svd(a)
    rot = ((vt.T)@(u.T)).T
    # check if we have found a reflection
    if np.linalg.det(rot) < 0:
        vt[2] = -vt[2]
        rot = ((vt.T)@(u.T)).T
    tran = av2 - av1@rot
    return tran, rot
