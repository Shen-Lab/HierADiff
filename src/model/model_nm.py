import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .diffusion_nm import FMdynamicBindModel
from .losses_nm import DynamicBindLoss

from .diffusion_flexFM_Rt import flexFM_RtScoreModel
from .losses_flexFM_Rt import flexFM_RtLoss

from .diffusion_flexSDE_Rt import flexSDE_RtScoreModel
from .losses_flexSDE_Rt import flexSDE_RtLoss

from .iRMSD_pre_model import IEGMN
from .losses_iRMSD_pred import compute_total_iRMSD_pred_loss

from .flexFM_Rt_confidence_model import flexFM_RtConfidenceEncoder
from .confidence_egnn import IEGMN_conf
from .losses_flexFM_Rt_confidence import flexFM_RtConfidenceLoss

# [FM_dynamicBindModel]
class FM_dynamicBindBaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(FM_dynamicBindBaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type # 'coord_flex_reg_model' now
        self.knn_size = args.knn_size # 20 defaulted
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = FMdynamicBindModel(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)



class FM_dynamicBindModel(FM_dynamicBindBaseModel):
    def __init__(self, args, params):
        super(FM_dynamicBindModel, self).__init__(args, params)
        # loss function
        self.loss = DynamicBindLoss(args)

        self._init()

    def forward(self, batch):
        # move graphs to cuda
        coord_r_pred, coord_l_pred, lddt_pred = self.encoder(batch)

        outputs = {}
        outputs["coord_r_pred"] = coord_r_pred 
        outputs["coord_l_pred"] = coord_l_pred
        outputs["lddt_pred"] = lddt_pred

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses
    

# [flexFM_RtModel]
class flexFM_RtBaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(flexFM_RtBaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type # 
        self.knn_size = args.knn_size # 20 defaulted
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = flexFM_RtScoreModel(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)



class flexFM_RtModel(flexFM_RtBaseModel):
    def __init__(self, args, params):
        super(flexFM_RtModel, self).__init__(args, params)
        # loss function
        self.loss = flexFM_RtLoss(args)
        self.args = args

        self._init()

    def forward(self, batch):
        if self.args.res_rot_flex:
            coord_r_pred, coord_l_pred, lddt_pred, rot_pred, tr_pred, delta_rec_rot, delta_lig_rot = self.encoder(batch)

            outputs = {}
            outputs["coord_r_pred"] = coord_r_pred 
            outputs["coord_l_pred"] = coord_l_pred
            outputs["lddt_pred"] = lddt_pred
            outputs["rot_pred"] = rot_pred
            outputs["tr_pred"] = tr_pred
            outputs["rec_res_rot_pred"] = delta_rec_rot
            outputs["lig_res_rot_pred"] = delta_lig_rot
        
        else:
            coord_r_pred, coord_l_pred, lddt_pred, rot_pred, tr_pred = self.encoder(batch)

            outputs = {}
            outputs["coord_r_pred"] = coord_r_pred 
            outputs["coord_l_pred"] = coord_l_pred
            outputs["lddt_pred"] = lddt_pred
            outputs["rot_pred"] = rot_pred
            outputs["tr_pred"] = tr_pred
        

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses



# [flexSDE_RtModel]
class flexSDE_RtBaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(flexSDE_RtBaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type # 
        self.knn_size = args.knn_size # 20 defaulted
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = flexSDE_RtScoreModel(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)


class flexSDE_RtModel(flexSDE_RtBaseModel):
    def __init__(self, args, params):
        super(flexSDE_RtModel, self).__init__(args, params)
        # loss function
        self.loss = flexSDE_RtLoss(args)
        self.args = args

        self._init()

    def forward(self, batch):
        if self.args.res_rot_flex:
            coord_r_pred, coord_l_pred, lddt_pred, rot_pred, tr_pred, delta_rec_rot, delta_lig_rot = self.encoder(batch)

            outputs = {}
            outputs["coord_r_pred"] = coord_r_pred 
            outputs["coord_l_pred"] = coord_l_pred
            outputs["lddt_pred"] = lddt_pred
            outputs["rot_pred"] = rot_pred
            outputs["tr_pred"] = tr_pred
            outputs["rec_res_rot_pred"] = delta_rec_rot
            outputs["lig_res_rot_pred"] = delta_lig_rot
        
        else:
            coord_r_pred, coord_l_pred, lddt_pred, rot_pred, tr_pred = self.encoder(batch)

            outputs = {}
            outputs["coord_r_pred"] = coord_r_pred 
            outputs["coord_l_pred"] = coord_l_pred
            outputs["lddt_pred"] = lddt_pred
            outputs["rot_pred"] = rot_pred
            outputs["tr_pred"] = tr_pred
        

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        return losses


# [iRMSD_predModel]
class iRMSD_predBaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(iRMSD_predBaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type # 
        self.knn_size = args.knn_size # 20 defaulted
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        self.encoder = IEGMN(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() < 2:
                continue
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)


class iRMSD_predModel(iRMSD_predBaseModel):
    def __init__(self, args, params):
        super(iRMSD_predModel, self).__init__(args, params)
        # loss function
        self.loss = compute_total_iRMSD_pred_loss(args)
        self.args = args

        self._init()

    def forward(self, batch):
        outputs = self.encoder(batch)

        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(outputs, batch)
        return losses

# [flexFM_RtConfidenceModel]
class flexFM_RtConfidenceBaseModel(nn.Module):
    """
        enc(receptor) -> R^(dxL)
        enc(ligand)  -> R^(dxL)
    """
    def __init__(self, args, params):
        super(flexFM_RtConfidenceBaseModel, self).__init__()

        ######## unpack model parameters
        self.model_type = args.model_type # 
        self.knn_size = args.knn_size # 20 defaulted
        self.args = args

        ######## initialize (shared) modules
        # raw encoders
        if args.model_type == "confidence_egnn":
            self.encoder = IEGMN_conf(args, params)
        else:
            self.encoder = flexFM_RtConfidenceEncoder(args, params)

        self._init()

    def _init(self):
        for name, param in self.named_parameters():
            # NOTE must name parameter "bert"
            if "bert" in name:
                continue
            # bias terms
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            elif param.dim() < 2:
                continue
            # weight terms
            else:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        raise Exception("Extend me")

    def dist(self, x, y):
        if len(x.size()) > 1:
            return ((x-y)**2).sum(-1)
        return ((x-y)**2)

class flexFM_RtConfidenceModel(flexFM_RtConfidenceBaseModel):
    def __init__(self, args, params):
        super(flexFM_RtConfidenceModel, self).__init__(args, params)
        # loss function
        self.loss = flexFM_RtConfidenceLoss(args)
        self.args = args

        self._init()
        # Cache the parameter list to avoid re-calling self.parameters() later.
        self._param_list = list(self.parameters())


    def forward(self, batch):
        
        if self.args.confidence_w_iRMSD_classification:
            lddt_pred, iRMSD_classification = self.encoder(batch)

            outputs = {}
            outputs["lddt_pred"] = lddt_pred
            outputs["iRMSD_classification"] = iRMSD_classification
        else:
            lddt_pred = self.encoder(batch)

            outputs = {}
            outputs["lddt_pred"] = lddt_pred

        if self.args.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
            outputs["l1_norm"] = l1_norm
        
        if self.args.l2_lambda > 0:
            l2_norm = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)
            outputs["l2_norm"] = l2_norm


        return outputs

    def compute_loss(self, batch, outputs):
        losses = self.loss(batch, outputs)
        # l1_lambda = self.args.l1_lambda
        # # Use the cached parameter list here.
        # l1_norm = sum(p.abs().sum() for p in self._param_list if p.requires_grad)
        # # l1_norm = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
        # losses["loss"] = losses["loss"] + l1_lambda * l1_norm
        return losses
