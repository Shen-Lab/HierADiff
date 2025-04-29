import os
import sys
import glob
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.data_parallel import DataParallel

from utils import printt, get_model_path
from .model_nm import FM_dynamicBindModel, flexFM_RtModel, flexSDE_RtModel, iRMSD_predModel, flexFM_RtConfidenceModel
from notebooks.utils_notebooks import Dict2Class

'''
The difference bt load_model and load_model_for_training is that the former is used for loading the model for inference, while the latter is used for loading the model for training.
the latter manages the epoch and optimizer as well.
'''

def load_model_nm(args, model_params, fold, load_best=True, confidence_mode = False):
    """
        Model factory
        :load_best: if True, load best model in terms of performance on val set, else load last model
    """
    # load model args
    
    with open(os.path.join(args.filtering_model_path if confidence_mode else args.score_model_path,"../args.yaml")) as f:
        model_args = yaml.safe_load(f)
    model_args = Dict2Class(model_args)
    model_args.gpu = args.gpu
    model_args.num_gpu = args.num_gpu


    # load model with specified arguments
    kwargs = {}
    if args.model_type == "FM_dynamicBind":
        model = FM_dynamicBindModel(model_args, model_params, **kwargs)
    elif args.model_type == "flexFM_Rt":
        model = flexFM_RtModel(model_args, model_params, **kwargs)
    elif args.model_type == "flexSDE_Rt":
        model = flexSDE_RtModel(model_args, model_params, **kwargs)
    elif args.model_type == "iRMSD_pred":
        model = iRMSD_predModel(model_args, model_params, **kwargs)
    elif "confidence" in args.model_type:
        model = flexFM_RtConfidenceModel(model_args, model_params, **kwargs)
    else:
        raise Exception(f"invalid model type {args.model_type}")
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))

    checkpoint=None
    # (optional) load checkpoint if provided
    if confidence_mode and args.filtering_model_path is not None:
        load_best = True # TODO: Remove
        if load_best:
            checkpoint = select_model(args.filtering_model_path,True)
        else:
            checkpoint = os.path.join(args.filtering_model_path, "model_last.pth")
    elif not confidence_mode and args.score_model_path is not None:
        if load_best:
            checkpoint = select_model(args.score_model_path,False)
        else:
            checkpoint = os.path.join(args.score_model_path, "model_last.pth")
    elif args.checkpoint_path is not None and os.path.exists(os.path.join(os.path.join(args.checkpoint_path, f"fold_{fold}"), "model_last.pth")):
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        if load_best:
            checkpoint = get_model_path(fold_dir)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")
    
    print("checkpoint",checkpoint)
      
    if checkpoint is not None:
        # extract current model
        state_dict = model.state_dict()
        # load onto CPU, transfer to proper GPU
        pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
        # update current model
        state_dict.update(pretrain_dict)
        # >>>
        for k,v in state_dict.items():
            if k not in pretrain_dict:
                print(k, "not saved")

        # DDP
        map_location = "cuda:%d" % args.local_rank
        model.load_state_dict(state_dict, map_location=map_location, strict=False)
        # model.load_state_dict(state_dict)
        printt("loaded checkpoint from", checkpoint)
    else:
        printt("no checkpoint found")

    return model

def select_model(fold_dir,confidence_mode):
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)

    if confidence_mode:
        models = sorted(paths, key=lambda s:-float(s.split("/")[-1].split("_")[-1][:-4]))
    else:
        models = sorted(paths, key=lambda s:float(s.split("/")[-1].split("_")[4]))

    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint

# [nm]
def load_model_for_training_nm(args, model_params, fold, load_best=True, confidence_mode = False, model_pth = None):
    """
        Model factory
        :load_best: if True, load best model in terms of performance on val set, else load last model
        :model_pth: if provided, load model from this path
    """
    # load model with specified arguments
    kwargs = {}
    if args.model_type == "FM_dynamicBind":
        model = FM_dynamicBindModel(args, model_params, **kwargs)
    elif args.model_type == "flexFM_Rt":
        model = flexFM_RtModel(args, model_params, **kwargs)
    elif args.model_type == "flexSDE_Rt":
        model = flexSDE_RtModel(args, model_params, **kwargs)
    elif args.model_type == "iRMSD_pred":
        model = iRMSD_predModel(args, model_params, **kwargs)
    elif "confidence" in args.model_type:
        model = flexFM_RtConfidenceModel(args, model_params, **kwargs)
    else:
        raise Exception(f"invalid model type {args.model_type}")
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))


    # !!!!
    print(model_pth, "!!!!!!")

    if model_pth is not None:
        if os.path.exists(model_pth):
            # extract current model
            state_dict = model.state_dict()
            # load onto CPU, transfer to proper GPU
            pretrain_dict = torch.load(model_pth, map_location="cpu")["model"]
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
            # update current model
            state_dict.update(pretrain_dict)
            # >>>
            for k,v in state_dict.items():
                if k not in pretrain_dict:
                    print(k, "not saved")
            model.load_state_dict(state_dict, strict=False)
            printt("loaded checkpoint from", model_pth)

            return model
        else:
            printt(f"no checkpoint from path {model_pth} found")

    # (optional) load checkpoint if provided and model_last.pth exists
    if args.checkpoint_path is not None and os.path.exists(os.path.join(os.path.join(args.checkpoint_path, f"fold_{fold}"), "model_last.pth")):
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        if load_best:
            checkpoint = select_model(fold_dir, confidence_mode=confidence_mode)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")
            
        if checkpoint is not None:
            # extract current model
            state_dict = model.state_dict()
            # load onto CPU, transfer to proper GPU
            checkpoint_dict = torch.load(checkpoint, map_location="cpu")
            pretrain_dict = checkpoint_dict["model"]
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
            # update current model
            state_dict.update(pretrain_dict)
            # >>>
            for k,v in state_dict.items():
                if k not in pretrain_dict:
                    print(k, "not saved")
            model.load_state_dict(state_dict, strict=False)
            printt("loaded checkpoint from", checkpoint)

            # load epoch to begin with; args.epoch only for training defaulted as 0
            args.epoch = checkpoint_dict["epoch"]

            # load optimizer before the training starts; outside of this func

        else:
            printt("no checkpoint found")
    return model

# [nm]
def to_cuda_nm(model, args):
    """
        move model to cuda
    """
    # specify number in case test =/= train GPU
    if args.gpu >= 0:
        # model = model.cuda(args.gpu)
        
        # DDP
        model = model.to(args.device)

        if args.num_gpu > 1:
            device_ids = [args.gpu + i for i in range(args.num_gpu)]
            model = DataParallel(model, device_ids=device_ids)
    return model


