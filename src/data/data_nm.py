import os
import sys
import copy
import pickle
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, DataListLoader

from tqdm import tqdm

#
from . import data_train_utils_nm as data_utils
#
from utils import compute_rmsd
#
from geom_utils import NoiseTransform_nm, set_time_nm, sample_vec, axis_angle_to_matrix, NoiseTransform_flexFM_Rt, NoiseTransform_flexSDE_Rt, Transform_iRMSD_pred, NoiseTransform_flexFM_Rt_confidence
from scipy.spatial.transform import Rotation as R

from utils import load_csv
from datetime import datetime
import random
import math

import traceback

import pandas as pd


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect any numpy._core.* → numpy.core.*
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)



# [self]
def set_random_rs_based_on_walltime():
    # randomly select a number according to the clock time
    current_time = datetime.now()
    current_millisecond = current_time.microsecond // 1000
    random.seed(current_millisecond)


# [self]
def shuffle_data_evenly(dict_pdb_to_totalLen, data, pdb_ids, args):
    '''shuffle data to ensure every mini batch has at most 1 long complex (args.threshold_total_len)
    Input:
        1. data (list): the list contains path to each ind pickle file
        2. pdb_ids (list): the list contains each complex id
        3. args (args class): the args class
    Output:
        1. shuffled_data (list): the shuffled list contains path to each ind pickle file
        2. shuffled_pdb_ids (list): the shuffled list contains each complex id
    '''

    #
    # filter out the complexes that are too long
    if args.filter_len_r_plus_l_max != 0: # 0 means no filter
        data, pdb_ids = [data[i] for i, pdb_id in enumerate(pdb_ids) if dict_pdb_to_totalLen[pdb_id] < args.filter_len_r_plus_l_max], [pdb_id for i, pdb_id in enumerate(pdb_ids) if dict_pdb_to_totalLen[pdb_id] < args.filter_len_r_plus_l_max]


    #
    # constr list of total complex lens
    list_totalLen = [dict_pdb_to_totalLen[i] for i in pdb_ids]

    paired = zip(list_totalLen, list(range(len(list_totalLen))))
    sorted_pairs = sorted(paired)
    sorted_index_list = [index for _, index in sorted_pairs]

    sorted_data = [data[index] for index in sorted_index_list]
    sorted_pdb_ids = [pdb_ids[index] for index in sorted_index_list]


    index_list = list(range(len(list_totalLen)))
    
    # print('sorted_data[:20]', sorted_data[:20])
    # print(aaaaa)

    #
    # sample bs-1 from the first sample_range datas     for small samples in each batch
    # use the last data for     large samples in each batch
    sample_range = 2*args.batch_size

    totalN = len(index_list)
    totalBatchN = totalN//args.batch_size + 1

    large_samp_index = []
    small_samp_index = []


    for i in range(totalBatchN):
        if not index_list:
            print("in the break logic")
            break
            
        large_samp_index.append(index_list[-1])
        del index_list[-1]
        
        sampled_index = random.sample(list(range(min(len(index_list),sample_range))), min(args.batch_size-1, len(index_list)))
        
        for ind in sampled_index:
            small_samp_index.append(index_list[ind])
        for ind in sorted(sampled_index, reverse=True):
            del index_list[ind]
    
    # print('large_samp_index', large_samp_index, 'small_samp_index', small_samp_index)
    # print(aaaaa)

    #
    # distribute large_overall_ind_list into overall index of the whole dataset
    large_overall_ind_list = []

    # x, y (relative)
    x_step = args.world_size
    # [i * step, (i+1) * step] for i in (0 to args.world_size - 1)

    # print("large_ind_arr", large_ind_arr)

    for i in range(len(large_samp_index)):
        # x (which group)
        x = i % x_step 
        # y (what position in each group)
        y = i // x_step * args.batch_size
        # index (absolute)
        abs_idx = x_step * y + x

        # print("args.batch_size", args.batch_size, "args.world_size", args.world_size)
        # print("large_ind_arr.shape[0]", large_ind_arr.shape[0], "x", x, "y", y, "abs_idx", abs_idx, "x_step", x_step)
        # print(aaaaa)

        large_overall_ind_list.append(abs_idx)

    #
    # distribute small_overall_ind_list into overall index of the whole dataset
    small_overall_ind_list = []

    small_overall_ind_list = list(set(list(range(len(data)))) - set(large_overall_ind_list))

    # print("small_overall_ind_list[:32]", small_overall_ind_list[:32])
    # print(aaaaa)


    # # x, y (relative)
    # x_step = args.world_size
    # # [i * step, (i+1) * step] for i in (0 to args.world_size - 1)

    # # print("large_ind_arr", large_ind_arr)

    # for i in range(len(small_samp_index)):
    #     # x (which group)
    #     x = (i //(args.batch_size-1)) % x_step 
    #     # y (what position in each group)
    #     y = (i //(args.batch_size-1)) // x_step * args.batch_size + (i %(args.batch_size-1))+1
    #     # index (absolute)
    #     abs_idx = x_step * y + x


    #     # print("args.batch_size", args.batch_size, "args.world_size", args.world_size)
    #     # print("x", x, "y", y, "abs_idx", abs_idx, "x_step", x_step, "i", i)
    #     # print(aaaaa)

    #     small_overall_ind_list.append(abs_idx)


    
    # print("large_overall_ind_list[:32]", large_overall_ind_list[:32], "small_overall_ind_list[:32]", small_overall_ind_list[:32])
    # print(aaaa)
    #
    shuffled_data = ['']*len(data)
    shuffled_pdb_ids = ['']*len(pdb_ids)

    # large
    for idx, value in zip(large_overall_ind_list, [sorted_data[i] for i in large_samp_index]):
        shuffled_data[idx] = value
    for idx, value in zip(large_overall_ind_list, [sorted_pdb_ids[i] for i in large_samp_index]):
        shuffled_pdb_ids[idx] = value
    
    # print("shuffled_data", shuffled_data, "shuffled_pdb_ids", shuffled_pdb_ids)

    # print("max(small_overall_ind_list)", max(small_overall_ind_list), "len(data)", len(data))
    # print("len(small_overall_ind_list)", len(small_overall_ind_list), "len(large_overall_ind_list)", len(large_overall_ind_list))

    # small
    for idx, value in zip(small_overall_ind_list, [sorted_data[i] for i in small_samp_index]):
        shuffled_data[idx] = value
    for idx, value in zip(small_overall_ind_list, [sorted_pdb_ids[i] for i in small_samp_index]):
        shuffled_pdb_ids[idx] = value
    

    # #
    # # random shuffled ind arrs
    # small_ind_arr = np.arange(0, len(data))[np.array(list_totalLen)< args.threshold_total_len]
    # large_ind_arr = np.arange(0, len(data))[np.array(list_totalLen) >= args.threshold_total_len]

    # np.random.shuffle(small_ind_arr)
    # np.random.shuffle(large_ind_arr)


    # #
    # # distribute large_ind_arr into overall index of the whole dataset
    # large_overall_ind_list = []

    # # x, y (relative)
    # x_step = args.world_size
    # # [i * step, (i+1) * step] for i in (0 to args.world_size - 1)

    # # print("large_ind_arr", large_ind_arr)

    # for i in range(large_ind_arr.shape[0]):
    #     # x (which group)
    #     x = i % x_step 
    #     # y (what position in each group)
    #     y = i // x_step * args.batch_size
    #     # index (absolute)
    #     abs_idx = x_step * y + x

    #     # print("args.batch_size", args.batch_size, "args.world_size", args.world_size)
    #     # print("large_ind_arr.shape[0]", large_ind_arr.shape[0], "x", x, "y", y, "abs_idx", abs_idx, "x_step", x_step)
    #     # print(aaaaa)

    #     large_overall_ind_list.append(abs_idx)
    
    # small_overall_ind_list = list(set(list(range(len(data)))) - set(large_overall_ind_list))

    # print("large_overall_ind_list", large_overall_ind_list, "small_overall_ind_list", small_overall_ind_list)
    # print(aaaaa)

    # shuffled_data = ['']*len(data)
    # shuffled_pdb_ids = ['']*len(pdb_ids)

    # # large
    # for idx, value in zip(large_overall_ind_list, [data[i] for i in large_ind_arr]):
    #     shuffled_data[idx] = value
    # for idx, value in zip(large_overall_ind_list, [pdb_ids[i] for i in large_ind_arr]):
    #     shuffled_pdb_ids[idx] = value
    
    # # print("shuffled_data", shuffled_data, "shuffled_pdb_ids", shuffled_pdb_ids)

    # # small
    # for idx, value in zip(small_overall_ind_list, [data[i] for i in small_ind_arr]):
    #     shuffled_data[idx] = value
    # for idx, value in zip(small_overall_ind_list, [pdb_ids[i] for i in small_ind_arr]):
    #     shuffled_pdb_ids[idx] = value


    # print("shuffled_data", shuffled_data, "shuffled_pdb_ids", shuffled_pdb_ids)
    # print(aaaaa)

    
    return shuffled_data, shuffled_pdb_ids
    












def load_data_nm(args, split=None, batch=None, verbose=True):
    """
        Load and minimally process data

        Args:
            split: If none, all splits are loaded. Otherwise, only the specified split.
            batch: If none, all batches are loaded. Otherwise, only the specified batch.
    """
    # add others like db5 or sabdab if we run on those
    if args.dataset == "dips":
        data = data_utils.DIPSLoader(args, split=split, batch=batch, verbose=verbose)
    elif args.dataset == "db5":
        data = data_utils.DB5Loader(args)
    elif args.dataset == "toy":
        data = data_utils.ToyLoader(args, split=split, batch=batch, verbose=verbose)
    else:
        raise Exception("invalid --dataset", args.dataset)
    return data


def get_data_nm(dataset, fold_num, args, for_reverse_diffusion=False, num_samples=1, reg_model_train=True):
    # dataset is the first return value of get_loader
    """
        Convert raw data into DataLoaders for training.
    """
    if args.use_randomized_confidence_data:
        dataset_class = RandomizedConfidenceDataset
    else:
        dataset_class = BindingDataset_nm

    
    use_pose = type(dataset) is tuple
    if use_pose:
        dataset, poses = dataset
    
    # get_splits
    if args.mode == "train":
        splits = dataset.crossval_split(fold_num)
    else:
        splits = dataset.get_splits() # TODO

    for split, pdb_ids in splits.items():
        pdb_ids = []
        data_to_load = load_csv(args.data_file)
        for line in data_to_load:
            if line["split"] == split:
                pdb_ids.append(line["path"])

        if args.subset is not None:
            # read in a csv file with pdb ids, the pdb ids are in the 'path' column, get all the pdb ids in a list
            new_pdb_ids = []
            data_to_load = load_csv(args.subset)
            for line in data_to_load:
                if line["split"] == split and line["path"] in pdb_ids:
                    new_pdb_ids.append(line["path"])
            pdb_ids = new_pdb_ids
    
        randomRL = (split == "train") and (not for_reverse_diffusion)
        splits[split] = dataset_class(dataset.dict_pdb_to_totalLen, args, dataset.dict_pdb_to_pklPath, pdb_ids, apply_transform=not for_reverse_diffusion, randomRL=randomRL)
        
    if "reg" in args.model_type and reg_model_train:
        data_loaders = _get_loader(splits, args)
        return data_loaders


    # current reverse diffusion does NOT use DataLoader
    # if for_reverse_diffusion and args.model_type not regression model:
    if for_reverse_diffusion:
        return splits
    
    # convert to DataLoader
    data_loaders = _get_loader(splits, args)
    return data_loaders


def _get_loader(splits, args):
    """
        Convert lists into DataLoader
    """
    # current reverse diffusion does NOT use DataLoader
    if args.mode == "test":
        return splits
    
    # convert to DataLoader
    loaders = {}
    for split, data in splits.items():
        # account for test-only datasets
        if len(data) == 0:
            loaders[split] = []
            continue

        # do not shuffle val/test
        shuffle = (split == "train")

        # set proper DataLoader object (PyG)
        if torch.cuda.is_available() and args.num_gpu > 1:
            loader = DataListLoader
        else:
            loader = DataLoader

        
        if shuffle and args.shuffle: # args.shuffle default is False
            if args.weighted_sampling: # according to the pdb middle two codes frequency
                sampler = DistributedWeightedSampler(data, shuffle=True, replacement=True)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False)


        loaders[split] = loader(data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=False, # this will make sure that the last batch is not dropped; the last batch may be smaller than the batch size
                                pin_memory=True,
                                sampler=sampler)
        
    return loaders

# ------ DATASET -------

class BindingDataset_nm(Dataset):
    """
    Protein-protein binding dataset with lazy loading.
    """

    def __init__(self, dict_pdb_to_totalLen, args, data, pdb_ids=None, apply_transform=True, randomRL=False):

        super(BindingDataset_nm, self).__init__()

        # intialize as empty
        if data == {}:
            self.args = args
            self.apply_transform = apply_transform
            self.data = []
            self.pdb_ids = []
            self.length = 0
            self.randomRL = randomRL and args.randomRL
            return

        self.randomRL = randomRL and args.randomRL

        self.args = args
        self.apply_transform = apply_transform

        # Select a subset for a given split
        # Results: 1. data (dict): {pdb_id: path_to_dict({"graph": , })} ######!!!!!!!!!
        #          2. self.pdb_ids (list)
        
        if pdb_ids is not None:
            data = {k: v for k, v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)

        # Store reference paths to data (not loading actual data here)
        if isinstance(data, list):
            self.data = data
        else:
            self.data = list(data.values())


        # shuffle according to the data size (both for train and val)
        if not args.shuffle:
            self.data, self.pdb_ids = shuffle_data_evenly(dict_pdb_to_totalLen, self.data, self.pdb_ids, args)

        middleTwoCodes = [i[:2] for i in self.pdb_ids]
        middleTwoCodes=pd.Series(middleTwoCodes)
        self.weights = (1./middleTwoCodes.map(middleTwoCodes.value_counts())).tolist()

        self.length = len(self.data)

    def len(self):
        return self.length
    
    def __delitem__(self, idx):
        """
            Easier deletion interface.
        """

        del self.data[idx]

        self.length = len(self.data)
        self.len = len(self.data)



    def get(self, idx):
        """
        Create a graph object when requested (lazy loading).
        """

        try:
            # data = pickle.load(open(self.data[idx], "rb"))
            data = CompatUnpickler(open(self.data[idx], "rb")).load()

        except:
            print("in the get function")
            print("len(self.data)", len(self.data))
            print("idx", idx)
            print("self.data[idx]", self.data[idx])

        item = data["graph"]

        item.rec_res_name_b = data['receptor_b_seq']
        item.lig_res_name_b = data['ligand_b_seq']



        # Only perform random swapping for training data
        if self.randomRL and random.random() < 0.5:
            # Swap the content of receptor and ligand instead of swapping the keys
            for key in item['receptor'].keys():
                item['receptor'][key], item['ligand'][key] = item['ligand'][key], item['receptor'][key]
                item.rec_res_name_b, item.lig_res_name_b = item.lig_res_name_b, item.rec_res_name_b

            # Also swap the relevant edges between receptor and ligand, if present
            if ("receptor", "contact", "receptor") in item and ("ligand", "contact", "ligand") in item:
                item[("receptor", "contact", "receptor")], item[("ligand", "contact", "ligand")] = \
                    item[("ligand", "contact", "ligand")], item[("receptor", "contact", "receptor")]

        
        # Apply the transformation if needed

        if self.apply_transform:
            if "flexFM_Rt" == self.args.model_type:
                item = NoiseTransform_flexFM_Rt(self.args)(item)
            elif "flexSDE_Rt" == self.args.model_type:
                item = NoiseTransform_flexSDE_Rt(self.args)(item)
            elif "iRMSD_pred" == self.args.model_type:
                item = Transform_iRMSD_pred(self.args)(item)
            elif "confidence" in self.args.model_type:
                item = NoiseTransform_flexFM_Rt_confidence(self.args)(item)
            else:
                item = NoiseTransform_nm(self.args)(item)
        # elif "reg" in self.args.model_type:
        #     item = regression_data_transform(self.args)(item)

        return item

    def get_visualization_values(self, pdb_name=None, index=None):
        if index is not None:
            pass
        elif pdb_name is not None:
            index = self.pdb_ids.index(pdb_name)
        else:
            raise Exception("Either pdb_name or index should be given!")

        # data = pickle.load(open(self.data[index], "rb"))
        data = CompatUnpickler(open(self.data[index], "rb")).load()
        return data["visualization_values"]

    def set_graph(self, idx, new_graph):
        try:
            # data = pickle.load(open(self.data[idx], "rb"))
            data = CompatUnpickler(open(self.data[idx], "rb")).load()
            print("path,", self.data[idx])
            data["graph"] = new_graph
        except Exception as e:
            print(traceback.format_exc())
            print("Error: Cannot load the file.")
            print('self.data[idx]', self.data[idx])
            print(aaa)

        pickle.dump(data, open(self.data[idx], "wb"))

    def deepcopy(self):
        """
        Create a deep copy of the dataset.
        """
        new_dataset = copy.deepcopy(self)

        new_data = []

        for full_path in tqdm(self.data, desc='Inf Copying'):
            directory, filename = os.path.split(full_path)
            new_filename = filename + '_cp'
            new_path = os.path.join(directory, new_filename)
            shutil.copy2(full_path, new_path)

            new_data.append(new_path)

        new_dataset.data = new_data

        return new_dataset

############ Modified ############

class RandomizedConfidenceDataset(Dataset):
    """
        Protein-protein dataset of randomly perturbed ligand poses used to train a confidence model. (experimental)
    """
    def __init__(self, args, data, pdb_ids=None, apply_transform=False):
        super(RandomizedConfidenceDataset, self).__init__()
        self.args = args
        # select subset for given split
        if pdb_ids is not None:
            data = {k:v for k,v in data.items() if k in pdb_ids}
            self.pdb_ids = [k for k in data if k in pdb_ids]
        else:
            self.pdb_ids = list(data)
        # convert to PyTorch geometric objects upon GET not INIT
        self.data = list(data.values())
        self.length = len(self.data)

    def len(self):
        return self.length

    def __delitem__(self, idx):
        """
            Easier deletion interface. MUST update length.
        """
        del self.data[idx]
        self.len = len(self.data)

    def get(self, idx):
        """
            Create graph object to keep original object intact,
            so we can modify pos, etc.
        """
        item = self.data[idx]["graph"]
        # >>> fix this later no need to copy tensors only references
        data = copy.deepcopy(item)
        set_time(data, 0, 0, 0, 1)
        if np.random.rand() < 0.05:
            tr_s_max=5
            rot_s_max=0.2
        else:
            tr_s_max=2
            rot_s_max=0.1
        return self.randomize_position_and_compute_rmsd(data, tr_s_max=tr_s_max, rot_s_max=rot_s_max) # 2 and 0.1 yields almost 50% -> good. 2 and 0.2 yields 13%

    def set_graph(self, idx, new_graph):
        self.data[idx]["graph"] = new_graph

    def randomize_position_and_compute_rmsd(self, complex_graph, tr_s_max, rot_s_max):
        # randomize rotation
        original_pos = complex_graph["ligand"].pos
        center = torch.mean(original_pos, dim=0, keepdim=True)
        # one way to generate random rotation matrix
        # random_rotation = torch.from_numpy(R.random().as_matrix())
        
        # Another way
        rot_update = sample_vec(eps=rot_s_max)# * rot_s_max
        rot_update = torch.from_numpy(rot_update).float()
        random_rotation = axis_angle_to_matrix(rot_update.squeeze())
        
        # yet another way
        #x = np.random.randn(3)
        #x /= np.linalg.norm(x)
        #x *= rot_s_max
        #random_rotation = R.from_euler('zyx', x, degrees=True)
        #random_rotation = torch.from_numpy(random_rotation.as_matrix())
        pos = (original_pos - center) @ random_rotation.T.float()

        # random translation
        tr_update = torch.normal(0, tr_s_max, size=(1, 3))
        pos = pos + tr_update + center
        complex_graph["ligand"].pos = pos

        # compute rmsd
        rmsd = compute_rmsd(original_pos, pos)
        return complex_graph, rmsd
    



class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    A custom sampler that extends DistributedSampler and integrates weighted sampling.
    This sampler ensures that each distributed process samples a unique subset of the dataset,
    and samples are drawn according to specified weights.
    """
    def __init__(
        self,
        dataset,
        num_replicas = None,
        rank = None,
        shuffle = True,
        seed = 0,
        drop_last = False,
        replacement = True
    ) -> None:
        # Call the parent class initializer
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        # Store weights and replacement flag
        self.weights = torch.as_tensor(dataset.weights, dtype=torch.double)
        self.replacement = replacement
    
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            self.generator.manual_seed(self.seed + self.epoch)
            indices = torch.multinomial(
                self.weights, len(self.dataset), self.replacement, generator=torch.Generator()
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


