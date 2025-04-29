import os
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from args_nm import parse_args
from data import load_data_nm, get_data_nm
from model import load_model_for_training_nm, to_cuda_nm
from utils import printt, print_res, log, get_unixtime
from train_nm import train, evaluate, evaluate_pose
from helpers import WandbLogger, TensorboardLogger
from sample_nm import sample


from datetime import datetime

torch.autograd.set_detect_anomaly(True)



args = parse_args()

# DDP setting
import torch.distributed as dist

args.world_size = int(os.environ["WORLD_SIZE"])
args.rank = int(os.environ['RANK'])
args.local_rank = int(os.environ["LOCAL_RANK"])

args.device = torch.device('cuda', args.local_rank)

args.gpu = args.local_rank
args.num_gpu = 1 # ???

print(args.rank, '/', args.world_size, 'initialization')

torch.cuda.set_device(args.gpu)
dist.init_process_group(backend='nccl')



def gen_random_rs_based_on_walltime():
    # randomly select a number according to the clock time
    current_time = datetime.now()
    current_millisecond = current_time.microsecond // 1000
    return current_millisecond


def set_seed(seed, random_seed=True):
    if random_seed:
        seed = seed + gen_random_rs_based_on_walltime()
        print("seed: ", seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def main(args=None):

    with open("goodluck.txt") as f:
        for line in f:
            print(line, end="")

    if args is None:
        args = parse_args()

    torch.hub.set_dir(args.torchhub_path)


    # init wandb before too long data loading to avoid timeout error thrown by wandb
    
    if args.mode != "test" and args.num_folds==1:
        
        writer = None # init
        if args.rank==0 and not args.data_process:
            log_dir = os.path.join(args.tensorboard_path,
                                    args.run_name,
                                    get_unixtime())
            if args.logger == "tensorboard":
                writer = TensorboardLogger(log_dir=log_dir)
            elif args.logger == "wandb":
                writer = WandbLogger(project=args.project, 
                                    entity=args.entity, 
                                    name=args.run_name, 
                                    group=args.group,
                                    config=args,
                                    idfolder=args.save_path)
            else:
                raise Exception("Improper logger.")
    
    # 
    # data processing
    # dist.barrier()

    # training mode, dump args for reproducibility
    if args.mode != "test":
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok = True)
        # save args
        with open(args.args_file, "w+") as f:
            yaml.dump(args.__dict__, f)

    # load raw data
    data = load_data_nm(args)
    data_params = data.data_params
    printt("finished loading raw data")

    if args.data_process:
        return

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    # train mode: train model for args.fold different seeds
    # and evaluate at the end
    if args.mode == "train":
        # save scores
        test_scores = defaultdict(list)
        # try different seeds
        # current args.num_folds as 0 (defaulted; test/val set)
        for fold in range(args.num_folds):
            
            #### set up fold experiment
            set_seed(args.seed)
            # make save folder
            fold_dir = os.path.join(args.save_path, f"fold_{fold}")
            args.fold_dir = fold_dir
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir, exist_ok=True)
            printt("fold {} seed {}\nsaved to {}".format(fold, args.seed, fold_dir))
            
            if args.num_folds > 1:
                # writer = None # if args.rank != 0
                writer = None
                if args.rank == 0:
                    log_dir = os.path.join(args.tensorboard_path,
                                            args.run_name, str(fold),
                                            get_unixtime())
                    if args.logger == "tensorboard":
                        writer = TensorboardLogger(log_dir=log_dir)
                    elif args.logger == "wandb":
                        # wandb logging object
                        writer = WandbLogger(project=args.project, # in the init bash script, "DiffDock Tuning" -> "DiffDock_nm Tuning" 
                                            entity=args.entity, # in the config (specified by the NAME in bash script)
                                            name=args.run_name, # in the init bash script, If wandb is used, the name of the run.
                                            group=args.group, # defaulted as None, For wandb logger, a group to organize individual runs into a larger experiment.
                                            config=args, # the whole args object passed in 
                                            idfolder=args.save_path) # the folder to save the wandb id
                    else:
                        raise Exception("Improper logger.")
            
            dist.barrier()
                
            
            
            # load and convert data to DataLoaders
            loaders = get_data_nm(data, fold, args) 
            printt("finished loading loaders. Now loader for reverse diffusion:")
            loaders_for_reverse_diffusion = get_data_nm(data, fold, args, for_reverse_diffusion=True)
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model_for_training_nm(args, data_params, fold, load_best=False)#, load_best=True) # load last_model to continue training
            model = to_cuda_nm(model, args)

            print("before torch.nn.parallel.DistributedDataParallel")
            # DDP
            model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank], find_unused_parameters=True)
            printt("finished loading model")

            numel = sum([p.numel() for p in model.parameters()])
            printt('Model with', numel, 'parameters')

            #### run training loop
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            printt(f'len(train_loader): {len(train_loader)}')
            printt(f'len(val_loader): {len(val_loader)}')
            if args.debug and False:
                for train_data in train_loader:
                    print('train_data')
                    print(train_data["name"])
                    break
                for val_data in val_loader:
                    print('val_data')
                    print(val_data["name"])
            #return
            
            
            best_score, best_epoch, best_path = train(
                    train_loader, val_loader,
                    model, writer, fold_dir, args, loaders_for_reverse_diffusion)
            printt("finished training best epoch {} loss {:.3f}".format(
                    best_epoch, best_score))

            #### run eval loop
            if best_path is not None:
                model = load_model_for_training_nm(args, data_params, fold)
                model.load_state_dict(torch.load(best_path,
                    map_location="cpu")["model"])
                model = to_cuda_nm(model, args)

                # DDP
                model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank], find_unused_parameters=True)
            
                printt(f"loaded model from {best_path}")

            # eval on test set
            test_score = evaluate(loaders["test"], model, writer, args)
            test_score["fold"] = fold
            # add val for hyperparameter search
            val_score = evaluate(loaders["val"], model, writer, args)
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val
            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            printt("fold {}".format(fold))
            print_res(test_score)
            # set next seed
            args.seed += 1
            break # run single fold
            # end of fold ========

        printt(f"{args.num_folds} folds average") 
        print_res(test_scores)
        log(test_scores, args.log_file)
        # end of all folds ========

    # test mode: load up all replicates from checkpoint directory
    # and evaluate by sampling from reverse diffusion process
    elif args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        test_scores = defaultdict(list)
        for fold_dir in os.listdir(args.save_path):
            if "fold_" not in fold_dir:
                continue
            fold = int(fold_dir[5:])
            # load and convert data to DataLoaders

            loaders = get_data_nm(data, fold, args)
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model_for_training_nm(args, data_params, fold)
            model = to_cuda_nm(model, args)
            
            # DDP
            model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank], find_unused_parameters=True)
            
            printt("finished loading model")

            # run reverse diffusion process
            samples_test = sample(loaders["test"], model, args)
            samples_val = sample(loaders["val"], model, args)

            # test fold
            test_score = evaluate_pose(loaders["test"], samples_test)

            # add val for hyperparameter search
            val_score = evaluate_pose(loaders["val"], samples_val)
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val

            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            # end of fold ========

        printt(f"Average test/val performance")
        print_res(test_scores)
        log(test_scores, args.log_file, reduction=False)
        # end of all folds ========


if __name__ == "__main__":

    # [norm]
    if args.arcsinh_norm:
        args.nm_s_max = args.nm_s_max_arcsinh_norm

        args.tr_s_max = args.tr_s_max_arcsinh_norm

    # [interpolation of nm]
    if args.interpolation_nm:
        args.nm_s_max = 1.0
        args.nm_s_min = 0.01

    # [self]
    main(args=args)

