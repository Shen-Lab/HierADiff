import os
import shutil
import sys
import yaml
import random
import resource
from collections import defaultdict
import pickle

import numpy as np
import torch
import torch.nn as nn


from args_nm import parse_args
from data import load_data_nm, get_data_nm
from model import load_model_for_training_nm, to_cuda_nm
from utils import printt
from train_nm import evaluate_pose
#
from sample_nm import sample

#
from sample_flexFM_Rt import sample_flexFM_Rt

from sample_flexSDE_Rt import sample_flexSDE_Rt

#
from predict_nm import predict, predict_coord

from pathlib import Path
from tqdm import tqdm

# from data.utils import DIPSLoader

# 
from data import BindingDataset_nm
from evaluation.compute_rmsd_nm import evaluate_all_rmsds, evaluate_all_ori_rmsds
from evaluation.compute_rmsd_nm import RMSDComputer


DATA_CACHE_VERSION = "v1" # the version of the data cache [?]



args = parse_args()

# '''
# # DDP setting
# import torch.distributed as dist
# args.world_size = int(os.environ["WORLD_SIZE"])
# args.rank = int(os.environ['SLURM_PROCID'])
# args.local_rank = args.rank % int(os.environ['LOCAL_WORLD_SIZE'])

# n = torch.cuda.device_count() // int(os.environ['LOCAL_WORLD_SIZE'])
# device_ids = list(range(local_rank * n, (local_rank + 1) * n))
# '''



# # Old
# # DDP setting
# import torch.distributed as dist
# args.world_size = int(os.environ["WORLD_SIZE"])
# # args.rank = int(os.environ['SLURM_PROCID'])
# # elastic
# args.rank = int(os.environ['RANK'])

# # elastic: there is no corelation bt local_rank and rank!!!
# args.local_rank = int(os.environ["LOCAL_RANK"])
# # args.local_rank = args.rank % torch.cuda.device_count()
# # args.device = 'cuda:' + str(args.local_rank)

# args.device = torch.device('cuda', args.local_rank)

# args.gpu = args.local_rank
# args.num_gpu = 1

# # print("torch.cuda.device_count()!!!!!!!", torch.cuda.device_count())
# #

# print(args.rank, '/', args.world_size, 'initialization')

# # [Mod]
# torch.cuda.set_device(args.gpu)

# # dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
# # elastic
# dist.init_process_group(backend='nccl')

args.world_size = 1
args.rank = 0
args.local_rank = torch.cuda.current_device()

args.device = torch.device('cuda', args.local_rank)

args.gpu = args.local_rank
args.num_gpu = 1

torch.cuda.set_device(args.gpu)






def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args=None, model_pth=None, output_rmsd=False, output_coeffs=False, output_path=None, split="test", all_results=None, lddt_out=False, pkl_ind=None):

    # he he he
    with open("goodluck.txt") as f:
        for line in f:
            print(line, end="")

    # if args is None:
    # args = parse_args()

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
    # args.seed
    # args.seed = 7 # !!!
    set_seed(args.seed)

    printt("random seed set to", args.seed)

    # load data params
    # data_params = load_data(args, split="test", batch=0).data_params # ori
    data_params = load_data_nm(args).data_params
    print("Loaded the data params")

    args.subset = None
    printt("subset:", args.subset)

    # load data
    data = load_data_nm(args)
    loaders_for_reverse_diffusion = get_data_nm(data, fold, args, for_reverse_diffusion=True, reg_model_train=False)

    # get model and load checkpoint, if relevant
    model = load_model_for_training_nm(args, data_params, fold, load_best=False, model_pth=model_pth) # load last_model to continue training # !!!!!!!!!!!!!!!!!! load_best = False : use the latest model !!!!!!!!!!!!!!!!!!!
    model = to_cuda_nm(model, args)
    printt("finished loading model")

    # print("before torch.nn.parallel.DistributedDataParallel")
    # # DDP
    # model = torch.nn.parallel.DistributedDataParallel(model, [args.local_rank], find_unused_parameters=True)
    # printt("finished loading model")

    # Log number of parameters
    numel = sum([p.numel() for p in model.parameters()])
    printt('Model with', numel, 'parameters')




    DEBUG = True

                
    # run reverse diffusion process
    # only use subset to run reverse diffusion process [TODO: modify this]
    data_list_ori = loaders_for_reverse_diffusion[split] #!!!!!


    #
    # split the data_list into multiple parts by contruct different datasets
    # run inference on each part with one gpu

    num_replicas = args.world_size
    total_size = len(data_list_ori)
    indices = list(range(total_size))

    indices = indices[args.rank:total_size:num_replicas] # !!!!!

    # [????]
    # indices = indices[args.rank:total_size:num_replicas][-4:] # !!!!! # [322,307,168,377,110] # [378,332,3,5,10]
    # indices = [0] #!!!!

    # NM_20 lrg / sml
    # list_ind = [378,332,3,5,10]#[322,307,168,377,110]#
    # sliced_indices = indices[args.rank:total_size:num_replicas]
    # # Selecting multiple specific indices from the sliced list
    # indices = [sliced_indices[i] for i in list_ind if i < len(sliced_indices)] # !!!!!
    
    # # NM_100
    # list_ind = [203, 219,   2,  25,  39]#[370, 308, 301, 168, 113]#
    # sliced_indices = indices[args.rank:total_size:num_replicas]
    # # Selecting multiple specific indices from the sliced list
    # indices = [sliced_indices[i] for i in list_ind if i < len(sliced_indices)] # !!!!!
            
    data_list = BindingDataset_nm({}, args, {}, apply_transform=False)
    data_list.data = [data_list_ori.data[i] for i in indices]
    data_list.pdb_ids = list(data_list.data)
    data_list.length = len(data_list.data)



    if DEBUG:
        print('total_size', total_size, 'rank: ', args.rank)
        # print('indices', indices, 'rank: ', args.rank)
        pass


    if DEBUG:
        print('data_list', data_list, 'rank: ', args.rank)
        # print(aaa)
        pass


    


    epoch_num = model_pth.split('_')[-1].split('.')[0]

    for i in range(40, 41):
        args.num_steps = i
        args.actual_steps = i

        if args.model_type == "iRMSD_pred":
            samples_val = predict(data_list, model, args, # epoch=epoch,
                            visualize_first_n_samples=args.visualize_n_val_graphs,
                            visualization_dir=args.visualization_path)
        return 

        # meter = evaluate_all_rmsds(data_list, samples_val)
        meter = evaluate_all_rmsds(samples_val)

        # #
        # # gather the attributes of meter from multiple gpus (first convert them to tensors)

        # complex_rmsd_tensor = torch.tensor(meter.complex_rmsd_list).to(args.device)
        # ligand_rmsd_tensor = torch.tensor(meter.ligand_rmsd_list).to(args.device)
        # interface_rmsd_tensor = torch.tensor(meter.interface_rmsd_list).to(args.device)
        # recovery_rmsd_prec_tensor = torch.tensor(meter.recovery_rmsd_prec_list).to(args.device)

        # if DEBUG:
        #     # print("rank: ", args.rank, "complex_rmsd_tensor", complex_rmsd_tensor)
        #     pass






        # # first construct a meter object with the attributes from all gpus
        # complex_rmsd_tensor = all_gather(complex_rmsd_tensor, args)
        # ligand_rmsd_tensor = all_gather(ligand_rmsd_tensor, args)
        # interface_rmsd_tensor = all_gather(interface_rmsd_tensor, args)
        # recovery_rmsd_prec_tensor = all_gather(recovery_rmsd_prec_tensor, args)


        # then call summarize on the meter object (modify the next line to do this)

        # if args.rank == 0:

        # meter_all = RMSDComputer()

        # meter_all.complex_rmsd_list = complex_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
        # meter_all.ligand_rmsd_list = ligand_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
        # meter_all.interface_rmsd_list = interface_rmsd_tensor.cpu() # now tensor and ok; only used by the summarize method
        # meter_all.recovery_rmsd_prec_list = recovery_rmsd_prec_tensor.cpu() # now tensor and ok; only used by the summarize method


        if DEBUG:
            # print("rank: ", args.rank, "meter_all.complex_rmsd_list", meter_all.complex_rmsd_list)
            pass

        # dist.barrier() 


        ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized, recovery_rmsd_prec_summarized, recovery_interface_rmsd_prec_summarized = meter.summarize(verbose=True)

        if output_path is not None:
            if not os.path.exists(output_path+".txt"):
                with open(output_path+".txt", 'w') as f:
                    f.write(f"model_pth: {model_pth}\n")
                    f.write(f"complex_rmsd_summarized: {complex_rmsd_summarized}\n")
                    f.write(f"interface_rmsd_summarized: {interface_rmsd_summarized}\n")
                    f.write(f"recovery_rmsd_prec_summarized: {recovery_rmsd_prec_summarized}\n")
                    f.write(f"recovery_interface_rmsd_prec_summarized: {recovery_interface_rmsd_prec_summarized}\n")
                    f.write("\n")
                
            else:
                with open(output_path+".txt", 'a') as f:
                    f.write(f"model_pth: {model_pth}\n")
                    f.write(f"complex_rmsd_summarized: {complex_rmsd_summarized}\n")
                    f.write(f"interface_rmsd_summarized: {interface_rmsd_summarized}\n")
                    f.write(f"recovery_rmsd_prec_summarized: {recovery_rmsd_prec_summarized}\n")
                    f.write(f"recovery_interface_rmsd_prec_summarized: {recovery_interface_rmsd_prec_summarized}\n")
                    f.write("\n")


            if not os.path.exists(output_path+'.pkl'):
                with open(output_path+'.pkl', 'wb') as f:
                    epoch_num = model_pth.split('_')[-1].split('.')[0]
                    recovery_rmsd_prec_list, complex_rmsd_list, interface_rmsd_list, recovery_interface_rmsd_prec_list, names = meter.recovery_rmsd_prec_list, meter.complex_rmsd_list, meter.interface_rmsd_list, meter.recovery_interface_rmsd_prec_list, meter.names
                    rmsd_dict = {epoch_num: {'recovery_rmsd_prec_list': recovery_rmsd_prec_list, 'complex_rmsd_list': complex_rmsd_list, 'interface_rmsd_list': interface_rmsd_list, 'names': names, 'recovery_interface_rmsd_prec_list': recovery_interface_rmsd_prec_list}}
                    pickle.dump(rmsd_dict, f)
            else:
                with open(output_path+'.pkl', 'rb') as f:
                    rmsd_dict = pickle.load(f)
                with open(output_path+'.pkl', 'wb') as f:
                    epoch_num = model_pth.split('_')[-1].split('.')[0]
                    recovery_rmsd_prec_list, complex_rmsd_list, interface_rmsd_list, recovery_interface_rmsd_prec_list, names = meter.recovery_rmsd_prec_list, meter.complex_rmsd_list, meter.interface_rmsd_list, meter.recovery_interface_rmsd_prec_list, meter.names
                    rmsd_dict[epoch_num] = {'recovery_rmsd_prec_list': recovery_rmsd_prec_list, 'complex_rmsd_list': complex_rmsd_list, 'interface_rmsd_list': interface_rmsd_list, 'names': names, 'recovery_interface_rmsd_prec_list': recovery_interface_rmsd_prec_list}
                    pickle.dump(rmsd_dict, f)

        if lddt_out:
            return all_results

        if output_rmsd:
            return meter.recovery_rmsd_prec_list, meter.complex_rmsd_list, meter.interface_rmsd_list
        
        if output_coeffs:
            return cat_updates




    # if writer is not None: # this is gauranteed as rank==0
    #     for rd_key, rd_value in ligand_rmsd_summarized.items():
    #         writer.add_scalar(f"val_lig_rmsd_{rd_key}", rd_value, num_batches)
    #     for rd_key, rd_value in complex_rmsd_summarized.items():
    #         writer.add_scalar(f"val_complex_rmsd_{rd_key}", rd_value, num_batches)
    #     if recovery_rmsd_prec_summarized is not None:
    #         for rd_key, rd_value in recovery_rmsd_prec_summarized.items():
    #             writer.add_scalar(f"val_recovery_rmsd_prec_{rd_key}", rd_value, num_batches)


    # # save model if it improves rmsd
    # if ligand_rmsd_summarized['mean'] < best_metrics['rmsds_mean'] and ligand_rmsd_summarized['median'] < best_metrics['rmsds_median']:
    #     best_metrics['rmsds_mean'] = ligand_rmsd_summarized['mean']
    #     best_metrics['rmsds_median'] = ligand_rmsd_summarized['median']
    #     best_epoch = epoch
    #     best_loss = val_loss

    #     path_suffix = f"{num_batches}_{epoch}_{best_metrics['rmsds_mean']:.3f}_{best_metrics['rmsds_median']:.3f}.pth"
    #     # save model ONLY IF best
    #     best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
    #     save_model(model, args, optimizer, best_path, best_epoch) # [? rank==0?]

    #     # save best stats
    #     best_stats_path = os.path.join(fold_dir, "best_stats.pkl")
    #     save_stats(best_metrics, best_epoch, best_loss, best_path, best_stats_path)

    # dist.barrier()        



# # [self]
# def all_gather(q, args):
#     """
#     Gathers tensor arrays of different lengths across multiple gpus (only 1-D tensors are supported for now)
    
#     Parameters
#     ----------
#         q : tensor array
#         args : args including args.world_size
        
#     Returns
#     -------
#         torch.cat(all_q) : cat of list of gathered tensor arrays from all the gpus

#     """
#     ws = args.world_size


#     local_size = torch.tensor(q.size(), device=q.device)
#     all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
#     dist.all_gather(all_sizes, local_size)
#     max_size = max(all_sizes)

#     size_diff = max_size.item() - local_size.item()
#     if size_diff:
#         padding = torch.zeros(size_diff, device=q.device, dtype=q.dtype)
#         q = torch.cat((q, padding))

#     all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
#     dist.all_gather(all_qs_padded, q)
#     all_qs = []
#     for q, size in zip(all_qs_padded, all_sizes):
#         all_qs.append(q[:size])
#     return torch.cat(all_qs)


if __name__ == "__main__":

    

    
    model_pth = os.environ["SAMPLING_MODEL_PATH"]

    print(model_pth, "????")

    if "DATA_PATH" in os.environ:
        args.data_path = os.environ["DATA_PATH"]
        args.data_file = os.environ["DATA_FILE"]

    if "OUTPUT_PATH" in os.environ:
        output_path = os.environ["OUTPUT_PATH"]
    else:
        output_path = None

    if "SPLIT" in os.environ:
        split = os.environ["SPLIT"]
    else:
        split = "test"

    if "visualize_n_val_graphs" in os.environ:
        args.visualize_n_val_graphs = int(os.environ["visualize_n_val_graphs"])
    else:
        args.visualize_n_val_graphs = 0

    if "SEED" in os.environ:
        args.seed = int(os.environ["SEED"])

    # if "LDDT_OUT" in os.environ:
    #     if os.environ["LDDT_OUT"] == "True":
    #         lddt_out = True
    #         pkl_ind=int(os.environ["PICKLE_INDEX"])
    #     else:
    #         lddt_out = False
    # else:
    #     lddt_out = False
    
    main(args=args, model_pth = model_pth, output_path=output_path, split=split)
    
    RMSD_out = False

    # if not RMSD_out:
    #     # NO RMSDs outputs
    #     main(args=args, model_pth = model_pth, output_path=output_path, split=split)
    # else:

        # complex_rmsd_lists = []
        # interface_rmsd_lists = []
        # recovery_rmsd_prec_lists = []

        # for i in range(7, 12):
        #     args.seed = i
        #     recovery_rmsd_prec_list, complex_rmsd_list, interface_rmsd_list = main(args=args, model_pth = model_pth, output_rmsd=True)
        #     complex_rmsd_lists.append(complex_rmsd_list)
        #     interface_rmsd_lists.append(interface_rmsd_list)
        #     recovery_rmsd_prec_lists.append(recovery_rmsd_prec_list)
        
        # # get the min RMSD along the dim 0 for complex and interface RMSDs
        # complex_rmsd_lists = np.array(complex_rmsd_lists)
        # interface_rmsd_lists = np.array(interface_rmsd_lists)
        # recovery_rmsd_prec_lists = np.array(recovery_rmsd_prec_lists)
        # complex_rmsd_min = complex_rmsd_lists.min(axis=0)
        # interface_rmsd_min = interface_rmsd_lists.min(axis=0)
        # recovery_rmsd_prec_max = recovery_rmsd_prec_lists.max(axis=0)


        # # get the mean, median and std of the min RMSDs
        # complex_rmsd_min_mean = complex_rmsd_min.mean()
        # complex_rmsd_min_median = np.median(complex_rmsd_min)
        # complex_rmsd_min_std = complex_rmsd_min.std()
        # interface_rmsd_min_mean = interface_rmsd_min.mean()
        # interface_rmsd_min_median = np.median(interface_rmsd_min)
        # interface_rmsd_min_std = interface_rmsd_min.std()
        # recovery_rmsd_prec_max_mean = recovery_rmsd_prec_max.mean()
        # recovery_rmsd_prec_max_median = np.median(recovery_rmsd_prec_max)
        # recovery_rmsd_prec_max_std = recovery_rmsd_prec_max.std()
        
        # # percentage for recovery_rmsd_prec_max bigger than 0.01
        # recovery_rmsd_prec_max_1 = (recovery_rmsd_prec_max > 0.01).sum() / len(recovery_rmsd_prec_max)
        # recovery_rmsd_prec_max_2 = (recovery_rmsd_prec_max > 0.02).sum() / len(recovery_rmsd_prec_max)
        # recovery_rmsd_prec_max_5 = (recovery_rmsd_prec_max > 0.05).sum() / len(recovery_rmsd_prec_max)
        # recovery_rmsd_prec_max_10 = (recovery_rmsd_prec_max > 0.1).sum() / len(recovery_rmsd_prec_max)


        # print("recovery_rmsd_prec_max_mean", recovery_rmsd_prec_max_mean)
        # print("recovery_rmsd_prec_max_median", recovery_rmsd_prec_max_median)
        # print("recovery_rmsd_prec_max_std", recovery_rmsd_prec_max_std)
        # print("recovery_rmsd_prec_max_1", recovery_rmsd_prec_max_1)
        # print("recovery_rmsd_prec_max_2", recovery_rmsd_prec_max_2)
        # print("recovery_rmsd_prec_max_5", recovery_rmsd_prec_max_5)
        # print("recovery_rmsd_prec_max_10", recovery_rmsd_prec_max_10)

        
        # print("complex_rmsd_min_mean", complex_rmsd_min_mean)
        # print("complex_rmsd_min_median", complex_rmsd_min_median)
        # print("complex_rmsd_min_std", complex_rmsd_min_std)
        # print("interface_rmsd_min_mean", interface_rmsd_min_mean)
        # print("interface_rmsd_min_median", interface_rmsd_min_median)
        # print("interface_rmsd_min_std", interface_rmsd_min_std)

