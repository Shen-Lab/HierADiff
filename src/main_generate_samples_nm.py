import os
import shutil
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


from args import parse_args
from data import load_data_nm, get_data_nm
from model import load_model_nm, to_cuda_nm
from utils import printt
from train import evaluate_pose
from sample import sample
from pathlib import Path
from tqdm import tqdm

from data.utils import DIPSLoader

DATA_CACHE_VERSION = "v1" # the version of the data cache [?]



args = parse_args()

'''
# DDP setting
import torch.distributed as dist
args.world_size = int(os.environ["WORLD_SIZE"])
args.rank = int(os.environ['SLURM_PROCID'])
args.local_rank = args.rank % int(os.environ['LOCAL_WORLD_SIZE'])

n = torch.cuda.device_count() // int(os.environ['LOCAL_WORLD_SIZE'])
device_ids = list(range(local_rank * n, (local_rank + 1) * n))
'''


# Old
# DDP setting
import torch.distributed as dist
args.world_size = int(os.environ["WORLD_SIZE"])
# args.rank = int(os.environ['SLURM_PROCID'])
# elastic
args.rank = int(os.environ['RANK'])

# elastic: there is no corelation bt local_rank and rank!!!
args.local_rank = int(os.environ["LOCAL_RANK"])
# args.local_rank = args.rank % torch.cuda.device_count()
# args.device = 'cuda:' + str(args.local_rank)

args.device = torch.device('cuda', args.local_rank)

args.gpu = args.local_rank
args.num_gpu = 1

# print("torch.cuda.device_count()!!!!!!!", torch.cuda.device_count())
#

print(args.rank, '/', args.world_size, 'initialization')

# [Mod]
torch.cuda.set_device(args.gpu)

# dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
# elastic
dist.init_process_group(backend='nccl')




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args=None):
    # he he he
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")

    # if args is None:
    #     args = parse_args()

    # torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.torchhub_path)

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    fold = 0

    #### set up fold experiment
    # set_seed(args.seed)

    # load data params
    # data_params = load_data(args, split="test", batch=0).data_params # ori
    data_params = load_data_nm(args).data_params
    print("Loaded the data params")

    # get model and load checkpoint, if relevant
    model = load_model_nm(args, data_params, fold, load_best=False) # load last_model to continue training # !!!!!!!!!!!!!!!!!! load_best = False : use the latest model !!!!!!!!!!!!!!!!!!!
    model = to_cuda_nm(model, args)
    printt("finished loading model")

    # Log number of parameters
    numel = sum([p.numel() for p in model.parameters()])
    printt('Model with', numel, 'parameters')
    # Get cache directory


    ########### modified until here ###########

    for split in ("test"): #("test", "val", "train"):
        print(f"Inference for {split} split!")

        # get number of batches for the split dataset [TODO: get_n_batches() is not implemented]
        n_batches = DIPSLoader.get_n_batches()[split]
        # n_batches = get_n_batches(args, split)


        # Randomize batch indexes [TODO: to be removed]
        batch_indexes = list(range(n_batches))
        # get_random_indexes_ignore_seed(batch_indexes, args.seed)
        random.shuffle(batch_indexes)

        for batch_index in tqdm(batch_indexes):
            # Get current directory
            directory = f"{args.samples_directory}/{split}/batch-{batch_index}"

            # If directory exists and is not empty, continue
            if os.path.exists(directory):
                print(f"batch {batch_index} is already generated. Continue!")
                continue

            # Load batch
            data = load_data_nm(args, split=split, batch=batch_index, verbose=False)
            batch = get_data_nm(data, fold, args, for_reverse_diffusion=True)[split]
            if len(batch) == 0:
                print("Zero batch!")
                continue

            # If not, create directory and continue
            os.makedirs(directory, exist_ok=True)

            # run reverse diffusion process
            samples_multiple_iterations = []
            rmsd_multiple_iterations = []
            try:
                for i in range(args.generate_n_predictions):
                    iteration = sample(batch, model, args)
                    rmsd = evaluate_pose(batch, iteration)["rmsd"]

                    samples_multiple_iterations.append(iteration)
                    rmsd_multiple_iterations.append(rmsd)

            except Exception as e:
                printt("RuntimeError. ")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                shutil.rmtree(directory)
                continue

            # For some reason, there were some samples without complex_t
            if "complex_t" not in samples_multiple_iterations[0][0]:
                print(f"No complex_t! Batch: {batch_index}.")

            serialize(samples_multiple_iterations, rmsd_multiple_iterations, directory=directory)


def get_random_indexes_ignore_seed(batches, current_seed):
    batch_indexes = list(range(len(batches)))

    # Randomize batch indexes
    random.seed(None)
    random.shuffle(batch_indexes)
    random.seed(current_seed)
    return batch_indexes


def split_list_into_batches(data, batch_size):
    number_of_batches = len(data) // batch_size + int(len(data) % batch_size > 0)
    batches = [data[i: i * batch_size] for i in range(number_of_batches)]
    return batches


def serialize(samples_multiple_iterations, rmsd_multiple_iterations, directory="."):
    torch.save(samples_multiple_iterations[0], f"{directory}/first_iteration.pkl")
    ligand_positions = [[graph["ligand"].pos for graph in graphs] for graphs in samples_multiple_iterations]
    torch.save(ligand_positions, f"{directory}/ligand_positions.pkl")
    torch.save(rmsd_multiple_iterations, f"{directory}/rmsds.pkl")


if __name__ == "__main__":
    main(args=args)

