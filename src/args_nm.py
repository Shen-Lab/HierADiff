import os
import yaml
import argparse

import glob

from utils import printt


def parse_args():
    parser = argparse.ArgumentParser("FlexibleDocking for proteins")

    # [self]
    parser.add_argument("--enable_random_init_axis_in_inv_tr", action="store_true", default=False,
                        help="If set, enable random initialization of axis in inv_tr")
    
    # attention features

    # [allInterAttentionFeature]
    parser.add_argument("--allInterAttentionFeature", action="store_true", default=False,
                        help="If set, use attention features in the all modules for all the inter features")

    # [readOutInterAttentionFeature]
    parser.add_argument("--readOutInterAttentionFeature", action="store_true", default=False,
                        help="If set, use attention features in the output module for the inter features")
    

    # nm features in the FM

    # [use_nm_msf_feat]
    parser.add_argument("--use_nm_msf_feat", action="store_true", default=False,
                        help="If set, use nm msf node features in the FM model")
    # [msf_embed_dim]
    parser.add_argument("--msf_embed_dim", type=int, default=16,
                        help="The embedding dimension for the msf node features")

    # [use_nm_mode_correlation_feat]
    parser.add_argument("--use_nm_mode_correlation_feat", action="store_true", default=False,
                        help="If set, use nm mode correlation edge features in the FM model")
    # [nm_corr_embed_dim]
    parser.add_argument("--nm_corr_embed_dim", type=int, default=16,
                        help="The embedding dimension for the nm mode correlation edge features")


    
    # [recLigMismatchingInterLayer]
    parser.add_argument("--recLigMismatchingInterLayer", action="store_true", default=True,
                        help="If set, use the recLigMismatchingInterLayer model")

    # [secondLastLayerLigFlex]
    parser.add_argument("--secondLastLayerLigFlex", action="store_true", default=False,
                        help="If set, use the secondLastLayerLigFlex model; THIS SHOULD BE USED WITH recLigMismatchingInterLayer")
    # [secondLastLayerLigFlex_halfl1]
    parser.add_argument("--secondLastLayerLigFlex_halfl1", action="store_true", default=False,
                        help="If set, use the secondLastLayerLigFlex model and use half l1 features for flexing readout; THIS SHOULD BE USED WITH recLigMismatchingInterLayer")
    # [indReadoutFlex_sharedWeight]
    parser.add_argument("--indReadoutFlex_sharedWeight", action="store_true", default=False,
                        help="If set, use the flex individual read out layer with shared weight, THIS SHOULD BE USED WITH recLigMismatchingInterLayer")
    
    # [indReadoutFlex_noSharedWeight]
    parser.add_argument("--indReadoutFlex_noSharedWeight", action="store_true", default=False,
                        help="If set, use the flex individual read out layer with no shared weight, THIS SHOULD BE USED WITH recLigMismatchingInterLayer")
    
    # [interModuleNoSharedWeight]
    parser.add_argument("--interModuleNoSharedWeight", action="store_true", default=False,
                        help="If set, use the inter module with no shared weight, THIS SHOULD BE USED WITH recLigMismatchingInterLayer")
    
    # [ASB]
    parser.add_argument("--ASB", action="store_true", default=False,
                        help="If set, use Aligned SB model")
    
    # [attention]
    parser.add_argument("--attention", action="store_true", default=False,
                        help="If set, use attention model in the output module")


    # [deform]
    parser.add_argument("--interpolation_nm", action="store_true", default=False,
                        help="If set, use the interpolation to deform according to nm.")
                        
    parser.add_argument("--deform_l2", action="store_true", default=False,
                        help="If set, use the l2 loss instead of l1 for nm deformation.")
    
    # [intra_nm_vector]
    parser.add_argument("--intra_equi_nm_node_f", action="store_true", default=False,
                        help="If set, use the nm vectors in the intra message passing.")

    # [freq features]
    parser.add_argument("--expanded_edge_w_freq_features", action="store_true", default=False,
                        help="If set, use the nm freq features in the output module for edge features.")

    #
    # ddp settings
    parser.add_argument("--world_size",
                        type=int, default=0,
                        help="world size")
    parser.add_argument("--rank",
                        type=int, default=0,
                        help="rank")
    parser.add_argument("--local_rank",
                        type=int, default=-1,
                        help="local rank")
    parser.add_argument("--device",
                        type=str, default="cuda: 0",
                        help="device to use")
    
    # # wandb resume logging
    # parser.add_argument("--wandb_resume",
    #                 choices=["allow", "must"],
    #                 type=str, default="allow",
    #                 help="the choice of whether to resume wandb logging forcefully or not")

    # do training using a subset of data 
    parser.add_argument("--subset",
                    type=str, default=None,
                    help="csv file path of data to use for training")
    
    # train_label_pos_b_ori
    parser.add_argument("--train_label_pos_b_ori",
                    action="store_true",
                    default=False,
                    help="traning use the original bound structures")
    

    



    # configuration
    parser.add_argument("--debug",
                        type=bool, default=False,
                        help="Set flag true to load smaller dataset")
    parser.add_argument("--config_file",
                        type=str, default=None,
                        help="YAML file")
    parser.add_argument("--args_file",
                        type=str, default="args.yaml",
                        help="Dump arguments for reproducibility")
    parser.add_argument("--log_file",
                        type=str, default="results.yaml",
                        help="Save results here")
    parser.add_argument("--tuned_param",
                        type=str, default="",
                        help="Which hyperparamter to tune.")

    # ======== data ========
    # include path to pdb protein files, splits
    parser.add_argument("--data_file",
                        type=str, default="",
                        help="Includes path to PDB files, splits")
    parser.add_argument("--recache",
                        action="store_true",
                        help="Flag to recache the processed input. Does not include graph and lm cache.")
    parser.add_argument("--no_graph_cache",
                        action="store_true",
                        help="Flag to disable caching of graphs")
    parser.add_argument("--no_lm_cache",
                        action="store_true",
                        help="Flag to disable caching of ESM")

    parser.add_argument("--data_path",
                        type=str, default="",
                        help="Root to raw data directory")
    parser.add_argument("--pose_path",
                        type=str, default="data",
                        help="Root to raw test poses directory")
    parser.add_argument("--save_path",
                        type=str, default="",
                        help="Root to model checkpoints")
    parser.add_argument("--torchhub_path",
                        type=str, default="torchhub",
                        help="Root to torch hub cache")
    parser.add_argument("--tensorboard_path",
                        type=str, default="runs",
                        help="Tensorboard directory")

    # specify data loading parameters
    parser.add_argument("--dataset",
                        choices=["dips", "db5", "toy"],
                        type=str, default="db5",
                        help="")
    parser.add_argument("--use_unbound",
                        action="store_true",
                        help="Bound or unbound for DB5")

    parser.add_argument("--resolution", default="residue",
                        choices=["residue", "backbone", "atom"],
                        help="resolution of individual points")

    parser.add_argument("--use_orientation_features", action="store_true", default=False,
                        help="If set, use orientation features for the edges as in EquiDock")

    # data len filter
    parser.add_argument("--filter_len_r_or_l", type=int, default=0,
                        help="The length value that is used to filter out the really small complexes")

    parser.add_argument("--filter_len_r_plus_l_max", type=int, default=0,
                        help="The length value that is used to filter out the really big complexes")
    
    # ???
    parser.add_argument("--ind", type=int, default=0,
                        help="The ind")
    parser.add_argument("--totalInd", type=int, default=1,
                        help="The hard-coded totalind")


    parser.add_argument("--randomRL", action="store_true", default=False,
                        help="If set, use random RL for the data; only in the regression mode")
    
    

    # [nm]
    parser.add_argument("--data_process", action='store_true', default=False,
                        help='data processing mode')
    
    
    # data batching
    parser.add_argument("--threshold_total_len", type=int, default=1600,
                        help="The threshold to char a complex as a larger one")
    
    # data loading
    parser.add_argument("--max_poses",
                        type=int, default=100,
                        help="maximum number of poses to load for eval")

    parser.add_argument("--num_workers",
                        type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--batch_size",
                        type=int, default=10,
                        help="number of protein complexes per batch")
    # logging
    parser.add_argument("--logger",
                        type=str, choices=["wandb", "tensorboard"],
                        default="tensorboard",
                        help="Which logger to use, wandb or tensorboard")
    

    # data shuffle [self]
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="If set, shuffle the data randomly; otherwise, use the even order in the data file")
    parser.add_argument("--weighted_sampling", action="store_true", default=False,
                        help="If set, sample the data with the weights defined by the frequency of the PDB middle two codes")
    # [nm]
    # in the init bash script
    parser.add_argument("--run_name",
                        type=str, default=None,
                        help=("(optional) tensorboard folder, aka 'comment' "
                              "field. used for dispatcher. If wandb is used, the name of the run."))
    
    parser.add_argument("--wandb_run_name",
                        type=str, default=None,
                        help=("(optional) Run name for wandb "
                              "If not provided, then run_name will be used."))
    parser.add_argument("--log_frequency",
                        type=int, default=10,
                        help="log every [n] batches")
    # [nm]
    # in process_args, read in config file
    parser.add_argument("--project",
                        type=str, default=None,
                        help="For wandb logger, the name of the project where you're sending the new run.")
    # [nm]
    # in process_args, read in config file
    parser.add_argument("--entity",
                        type=str, default=None,
                        help="For wandb logger, a username or team name where you're sending runs.")
    # [nm]
    parser.add_argument("--group",
                        type=str, default=None,
                        help="For wandb logger, a group to organize individual runs into a larger experiment.")
    parser.add_argument("--visualize_n_val_graphs",
                        type=int, default=5,
                        help="How many of validation graphs should be visualized.")
    parser.add_argument("--visualization_path",
                        type=str, default="./visualization",
                        help="Where to save visualizations.")

    # data processing
    parser.add_argument('--receptor_radius', type=float, default=30,
                        help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10,
                        help='Max number of neighbors for each residue')
    parser.add_argument('--atom_radius', type=float, default=5,
                        help='Cutoff on distances for atom connections')
    parser.add_argument('--atom_max_neighbors', type=int, default=8,
                        help='Max number of atom neighbours for receptor')
    parser.add_argument('--matching_popsize', type=int, default=20,
                        help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20,
                        help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None,
                        help='Max number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=False,
                        help='remove Hs')
    parser.add_argument('--multiplicity', type=int, default=1, help='multiplicity parameter for debugging')

    # [nm]
    # data for nm
    parser.add_argument('--n_NM', type=int, default=20,
                        help='The number of first non-trivial normal modes')
    

    # ====== training ======
    parser.add_argument("--mode",
                        choices=["train", "test"],
                        type=str, default="train",
                        help="Training or inference")

    parser.add_argument("--num_folds",
                        type=int, default=5,
                        help="Number of different seeds = cv folds")
    parser.add_argument("--test_fold",
                        type=int, default=0,
                        help="Fold to use for inference")
    parser.add_argument("--epochs",
                        type=int, default=200,
                        help="Max epochs to train")
    parser.add_argument("--patience",
                        type=int, default=10,
                        help="Lack of validation improvement for [n] epochs")
    parser.add_argument("--metric",
                        type=str, default="loss",
                        help="For printing only")

    parser.add_argument("--gpu",
                        type=int, default=0,
                        help="GPU id")
    parser.add_argument("--num_gpu",
                        type=int, default=1,
                        help="number of GPU for DataParallel")
    parser.add_argument("--seed",
                        type=int, default=0,
                        help="Initial seed")

    parser.add_argument("--save_pred",
                        action="store_true",
                        help="Save predictions on test set")
    parser.add_argument("--no_tqdm",
                        action="store_true",
                        help="Set to True if running dispatcher")
    parser.add_argument("--save_model_every",
                        type=int, default=10,
                        help="Frequency (in epochs) of saving the latest model")

    # [self]
    parser.add_argument("--save_model_every_batch",
                        type=int, default=10,
                        help="Frequency (in batches) of saving the latest model")

    # ====== inference and sample ======
    parser.add_argument("--num_steps",
                        type=int, default=20,
                        help="Number of denoising steps")
    parser.add_argument("--actual_steps",
                        type=int, default=40,
                        help="Number of actual denoising steps. The intuition is to cut the last steps short because those tend to overfit.")
    parser.add_argument("--ode",
                        action="store_true",
                        help="Use ODE for inference")
    parser.add_argument("--no_random",
                        action="store_true",
                        help="Use no randomness in reverse diffusion")
    parser.add_argument("--no_final_noise",
                        action="store_true",
                        help="Use no noise in the final step of "
                             "reverse diffusion")
    parser.add_argument('--val_inference_freq',
                        type=int,
                        default=5,
                        help='Frequency of epochs for which to run expensive inference on val data')

    # [args.val_sample_flag]
    # [self]
    parser.add_argument('--val_sample_flag', 
                        action="store_true",
                        default=False,
                        help="If set, sample the val data")
    # [args.starting_from_random_flexed_unbound]
    # [self]
    parser.add_argument('--starting_from_random_flexed_unbound',
                        action="store_true",
                        default=False,
                        help="If set, start the sampling from unbound")

    
    parser.add_argument('--num_inference_complexes',
                        type=int,
                        default=None,
                        help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--num_inference_complexes_train_data',
                        type=int,
                        default=None,
                        help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument("--sample_train",
                        action="store_true",
                        help="Whether to run inference on training data. Useful for DB5.")


    # ======== model =======
    parser.add_argument("--epoch",
                        type=int, default=0,
                        help="epoch to be loaded")
    parser.add_argument("--model_type",
                        choices=["e3nn", "e3nn_inter", "gat", "time_pred"],
                        type=str, default="e3nn")
    parser.add_argument("--ebd_type",
                        choices=["continuous", "discrete"],
                        type=str, default="continuous")
    parser.add_argument("--encoder_type",
                        choices=["e3nn", "e3nn_inter", "gat", "time_pred"],
                        type=str, default="e3nn",
                        help="protein encoder")
    
    parser.add_argument("--layer_norm", action="store_true", default=False,
                        help="If not set, only linear layer is used for node embedding and no layer_norm for edge embedding; If set, MLP (relu, dropout, layernorm) is used for node embedding and layer_norm for edge embedding")
    
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5,
                        help="Epsilon for layer norm")
    
    #
    parser.add_argument("--max_num_cross_neighbors", type=int, default=10,
                        help="Max number of cross neighbors")
    

    # diffusion arcsinh norm
    parser.add_argument("--arcsinh_norm", action="store_true", default=False,
                        help="If set, the score is normalized by arcsinh; no y_limit")

    parser.add_argument("--nm_s_max_arcsinh_norm", type=float, default=1.0,
                        help="Max sigma for nm component when arcsinh norm is used")
    
    parser.add_argument("--tr_s_max_arcsinh_norm", type=float, default=3.4,
                        help="Max sigma for tr component when arcsinh norm is used; default as np.log(30)")
    
    parser.add_argument("--nm_update_min", type=float, default=-30,
                        help="Min value for nm update when arcsinh norm is used")
    
    parser.add_argument("--nm_update_max", type=float, default=30,
                        help="Max value for nm update when arcsinh norm is used")
    
    parser.add_argument("--tr_update_min", type=float, default=-90,
                        help="Min value for tr update when arcsinh norm is used")
    
    parser.add_argument("--tr_update_max", type=float, default=90,
                        help="Max value for tr update when arcsinh norm is used")
    
    parser.add_argument("--arcsinh_train_resample", action="store_true", default=False,
                        help="If set, resample the training data for arcsinh norm if the update is out of range; default as clamping")

    parser.add_argument("--no_range_limit", action="store_true", default=False,
                        help="If set, no using resampling or clamping for the update")

    


    

    # score model tanh norm
    parser.add_argument("--tanh_norm", action="store_true", default=False,
                        help="If set, the score is normalized by tanh; y_limit (-1, 1)")
    

    # (optional)
    parser.add_argument("--checkpoint_path",
                        type=str, default=None,
                        help="Checkpoint for entire model for test/finetune")

    # Embeddings
    parser.add_argument("--hidden_size",
                        type=int, default=64,
                        help="Hidden layer representation size")
    #
    parser.add_argument("--dropout",
                        type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--knn_size",
                        type=int, default=20,
                        help="max kNN size for edge construction")

    # E3NN
    #
    parser.add_argument("--num_conv_layers",
                        type=int, default=5,
                        help="Number of interaction layers")
    # 
    parser.add_argument("--max_radius",
                        type=float, default=5.,
                        help="Radius cutoff for geometric graph (A)")
    # 
    parser.add_argument("--intra_max_radius",
                        type=float, default=10.,
                        help="Radius cutoff for geometric graph for intra edges (A)")
    
    # 
    parser.add_argument("--rec_max_radius",
                        type=float, default=30.,
                        help="Radius cutoff for geometric graph for intra-rec edges; only used for gaussian smear")
    #
    parser.add_argument("--scale_by_sigma", action="store_true", default=True,
                        help="Whether to normalise the score")
    #
    parser.add_argument("--ns",
                        type=int, default=16,
                        help="Number of hidden features per node of order 0")
    #
    parser.add_argument("--nv",
                        type=int, default=4,
                        help="Number of hidden features per node of order >0")
    #
    parser.add_argument("--dist_embed_dim",
                        type=int, default=32,
                        help="Embedding size for the distance")
    #
    parser.add_argument("--cross_dist_embed_dim",
                        type=int, default=32,
                        help="Embeddings size for the cross distance")
    parser.add_argument("--lm_embed_dim",
                        type=int, default=0,
                        help="0 or 1280 for ESM2")
    
    # [use_x_saprot]
    parser.add_argument("--use_x_saprot", action="store_true", default=False,
                        help="If set, use the saprot features for node features, instead of the ESM2 features")
                        
    #
    parser.add_argument("--no_batch_norm", action="store_true", default=False,
                        help="If set, it removes the batch norm")
    #
    parser.add_argument("--use_second_order_repr",
                        action="store_true", default=False,
                        help="Whether to use only up to first order representations or also second")
    # 
    parser.add_argument("--cross_max_dist",
                        type=float, default=80,
                        help="Max cross distance in case not dynamic")
    # 
    parser.add_argument("--dynamic_max_cross",
                        action="store_true", default=False,
                        help="Whether to use the dynamic distance cutoff")
    #
    parser.add_argument("--cross_cutoff_weight",
                        type=float, default=3,
                        help="The weight that multpilies tr_s in case dynamic. Dynamic cross cutoff is of the form tr_s * weight + bias")
    #
    parser.add_argument("--cross_cutoff_bias",
                        type=float, default=40,
                        help="The bias that gets added to tr_s * weight in case dynamic. Dynamic cross cutoff is of the form tr_s * weight + bias")
    parser.add_argument("--embedding_type",
                        type=str, default="sinusoidal",
                        help="Type of diffusion time embedding")
    #
    parser.add_argument("--sigma_embed_dim",
                        type=int, default=32,
                        help="Size of the embedding of the diffusion time")
    parser.add_argument("--embedding_scale",
                        type=int, default=10000,
                        help="Parameter of the diffusion time embedding")
    
    # norm by tanh
    parser.add_argument("--norm_by_tanh", action="store_true", default=False,
                        help="If set, the score is normalized by tanh")

    # ==== optimization ====
    # loss type FAPE
    parser.add_argument("--FAPE_loss",
                        action="store_true", default=False,
                        help="If set, use FAPE loss")
    
    
    # loss term weights
    parser.add_argument("--score_loss_weight", default=0.,
                        type=float,
                        help="Loss weight for score matching")
    parser.add_argument("--energy_loss_weight", default=1.,
                        type=float,
                        help="Loss weight for energy margin")
    
    # flexFM_Rt

    # lateFlex
    parser.add_argument("--lateFlex", action="store_true", default=False,
                        help="If set, use flexing only after the specific epoch")
    parser.add_argument("--lateFlex_epoch", default=12,
                        type=int, help="The epoch to start flexing")
    

    # [zeroing_cutoff]
    parser.add_argument("--zeroing_cutoff", default=15.0,
                        type=float,
                        help="Zeroing out effect cutoff")

    parser.add_argument("--flex_tr_weight", default=1.,
                        type=float,
                        help="Loss weight for tr flexibility")
    parser.add_argument("--flex_rot_weight", default=1.,
                        type=float,
                        help="Loss weight for rot flexibility")
    # [flex_fape_weight]
    parser.add_argument("--flex_fape_weight", default=1.,
                        type=float,
                        help="Loss weight for fape loss")
    

    
    
    # rec_tr_weight, rec_rot_weight, lig_tr_weight, lig_rot_weight
    parser.add_argument("--rec_tr_weight", default=1.,
                        type=float,
                        help="Loss weight for receptor translation")
    parser.add_argument("--rec_rot_weight", default=1.,
                        type=float,
                        help="Loss weight for receptor rotation")
    parser.add_argument("--lig_tr_weight", default=1.,
                        type=float,
                        help="Loss weight for ligand translation")
    parser.add_argument("--lig_rot_weight", default=1.,
                        type=float,
                        help="Loss weight for ligand rotation")

                
    # lddt loss weight
    parser.add_argument("--lddt_weight", default=1.,
                        type=float,
                        help="Weight of lddt loss") 
    
    # [conf_time]
    parser.add_argument("--conf_time", default=0.9875,
                        type=float,
                        help="Time for confidence model")
    
    # [l1_lambda]
    parser.add_argument("--l1_lambda", default=0.,
                        type=float,
                        help="Weight of l1 norm")
    
    # [l2_lambda]
    parser.add_argument("--l2_lambda", default=0.,
                        type=float,
                        help="Weight of l2 norm")

    # [irmsd_confidence_weight]
    parser.add_argument("--irmsd_confidence_weight", default=1.,
                        type=float,
                        help="Weight of irmsd confidence loss")
    
    #[confidence_w_iRMSD_classification]
    parser.add_argument("--confidence_w_iRMSD_classification", action="store_true", default=False,
                        help="If set, use addtional iRMSD classification for the confidence loss")
    
    
    

    # complex RMSD loss weight (defaulted as 0.0)
    parser.add_argument("--complexRMSD_weight",
                        type=float, default=0.0,
                        help="Weight of complex RMSD loss")
    
    # interface RMSD loss weight (defaulted as 0.0)
    parser.add_argument("--interRMSD_weight",
                        type=float, default=0.0,
                        help="Weight of interface RMSD loss")

    # optimizer
    parser.add_argument("--lr",
                        type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay",
                        type=float, default=1e-6,
                        help="L2 regularization weight")

    # diffusion
    parser.add_argument("--tr_weight",
                        type=float, default=1.0,
                        help="Weight of translation loss")
    parser.add_argument("--rot_weight",
                        type=float, default=1.0,
                        help="Weight of rotation loss")
    # [nm]
    parser.add_argument("--nm_r_weight",
                        type=float, default=0.17,
                        help="Weight of r nm loss")
    parser.add_argument("--nm_l_weight",
                        type=float, default=0.17,
                        help="Weight of l nm loss")


    # flexFM_Rt # use rot_s_min and rot_s_max and tr_s_min and tr_s_max ALSO
    parser.add_argument("--flex_rec_s_min",
                        type=float, default=0.01,
                        help="Min sigma for rec flex component")
    parser.add_argument("--flex_rec_s_max",
                        type=float, default=1.0,
                        help="Max sigma for rec flex component")
    parser.add_argument("--flex_lig_s_min",
                        type=float, default=0.01,
                        help="Min sigma for lig flex component")
    parser.add_argument("--flex_lig_s_max",
                        type=float, default=1.0,
                        help="Max sigma for lig flex component")
    
    # [natural_exp]
    parser.add_argument("--natural_exp", action="store_true", default=False,
                        help="If set, use exponential scheduling")
    parser.add_argument("--c", default=10.0,
                        type=float,
                        help="power factor for natural exp flexing scheduling")
    
    # [adaptive_schedule]
    parser.add_argument("--adaptive_schedule", action="store_true", default=False,
                        help="If set, use adaptive c scheduling")

    # [SDE time scheduling -qln(1-t)]
    parser.add_argument("--sde_tSchedule_factor", default=1.0,
                        type=float,
                        help="multiplication factor q for SDE time scheduling -ln(1-t)")
    
    # [eigenscale] "log", "affine"
    parser.add_argument("--eigenscale", choices=["log", "affine"],
                        type=str, default="log",
                        help="The scheduling for the eigenvalue scaling")

    
    # [balanced_diffusion]
    parser.add_argument("--balanced_diffusion", action="store_true", default=False,
                        help="If set, use balanced diffusion term with a factor of sqrt(H)")

    # [neg_drift]
    parser.add_argument("--neg_drift", action="store_true", default=False,
                        help="If set, use negative drift in the reverse sampling")
    
    # [flexSDE_Rt] [flex_tr_t_schedule]
    parser.add_argument("--flex_tr_t_schedule", choices=["log", "linear", "square"],
                        type=str, default="linear",
                        help="The scheduling for the translation flexibility")
    




    # A_factor
    parser.add_argument("--A_factor", default=5.0,
                        type=float,
                        help="base scaling for flexing scheduling")
    # B_exp
    parser.add_argument("--B_exp", default=0.3,
                        type=float,
                        help="power factor for flexing scheduling")
    
    # nearest_res_decay
    parser.add_argument("--nearest_res_decay", default=False,
                        action="store_true",
                        help="If set, use the nearest residue distance for decay weight")
    
    

    parser.add_argument("--rot_s_min",
                        type=float, default=0.1,
                        help="Min sigma for rotational component")
    parser.add_argument("--rot_s_max",
                        type=float, default=1.65,
                        help="Max sigma for rotational component")
    parser.add_argument("--tr_s_min",
                        type=float, default=0.1,
                        help="Min sigma for translational component")
    parser.add_argument("--tr_s_max",
                        type=float, default=30,
                        help="Max sigma for translational component")
    

    #
    parser.add_argument("--tr_only", action="store_true", default=False,
                        help="If set only translational matching")

    parser.add_argument("--invariant_tr_output", action="store_true", default=False,
                        help="If set, the translation displacement output is invariant to the input")
    
    parser.add_argument("--norm_tr_nm_like_vec", action="store_true", default=False,
                        help="If set, the translation-nm-like vector is normalized like the nm vector")

    parser.add_argument("--diff_tr_weight", action="store_true", default=False,
                        help="If set, the translation inv output layer is not shared in weights")
    
    parser.add_argument("--expanded_edge_w_l0_features", action="store_true", default=False,
                        help="If set, the edge features are expanded with l0 features from nm and distance")
    


    # [nm]

    # in the flow-matching model (CI model)

    parser.add_argument("--tr_rec_s_min",
                        type=float, default=0.01,
                        help="Min sigma for rec tr component")
    parser.add_argument("--tr_rec_s_max",
                        type=float, default=1.0,
                        help="Max sigma for rec tr component")
    parser.add_argument("--rot_rec_s_min",
                        type=float, default=0.01,
                        help="Min sigma for rec rot component")
    parser.add_argument("--rot_rec_s_max",
                        type=float, default=1.0,
                        help="Max sigma for rec rot component")
    parser.add_argument("--tr_lig_s_min",
                        type=float, default=0.01,
                        help="Min sigma for lig tr component")
    parser.add_argument("--tr_lig_s_max",
                        type=float, default=1.0,
                        help="Max sigma for lig tr component")
    parser.add_argument("--rot_lig_s_min",
                        type=float, default=0.01,
                        help="Min sigma for lig rot component")
    parser.add_argument("--rot_lig_s_max",
                        type=float, default=1.0,
                        help="Max sigma for lig rot component")
    
    
    
    parser.add_argument("--nm_s_min",
                        type=float, default=0.01,
                        help="Min sigma for nm component")
    parser.add_argument("--nm_s_max",
                        type=float, default=10,
                        help="Max sigma for nm component")
    parser.add_argument("--no_nm", action="store_true", default=False,
                        help="If set only rigid matching")
    
    # [flex]
    parser.add_argument("--flex_only", action="store_true", default=False,
                        help="If set only flex matching")

    # res_rot_flex
    parser.add_argument("--res_rot_flex", action="store_true", default=False,
                        help="If set use residue level frame rot for flex")

       

    # confidence model
    parser.add_argument('--rmsd_prediction', action='store_true', 
                        default=False, 
                        help='If true, use regression against RMSD values')
    parser.add_argument('--rmsd_classification_cutoff', 
                        type=float, default=5, 
                        help='RMSD value below which a prediction is considered a postitive.')
    parser.add_argument("--generate_n_predictions", type=int, default=7,
                        help="For generating samples, how many predictions should be generated per samples.")
    parser.add_argument("--samples_directory", type=str, default="",
                        help="Directory in which sampels should be saved & loaded for confidence model.")
    # [sampling_model_pth]
    parser.add_argument("--sampling_model_pth", type=str, default="",
                        help="Path to the sampling model")
                        
    parser.add_argument('--use_randomized_confidence_data', action='store_true', 
                        default=False, 
                        help='If true, randomly generated data for training the confidence mdoel. Else, use data generated by score model')
    parser.add_argument('--filtering_model_path', 
                        default=None, type=str,
                        help='path to filtering model')
    parser.add_argument('--score_model_path',type=str, 
                        default=None, 
                        help='path to score model')
    
    parser.add_argument('--num_samples',type=int, 
                        default=None, 
                        help='number of samples in inference')

    parser.add_argument('--mirror_ligand',type=bool, 
                    default=False, 
                    help='number of samples in inference')
    
    parser.add_argument('--prediction_storage',type=str, 
                default=False, 
                help='output path to predictions')

    parser.add_argument("--temp_sampling", type=float, default=1.0, help="")
    parser.add_argument("--temp_psi", type=float, default=0.0, help="")
    parser.add_argument("--temp_sigma_data_tr", type=float, default=0.5, help="")
    parser.add_argument("--temp_sigma_data_rot", type=float, default=0.5, help="")
    # [nm]
    parser.add_argument("--temp_sigma_data_nm", type=float, default=0.5, help="")
                

    parser.add_argument('--run_inference_without_confidence_model', action='store_true', 
                        default=False, 
                        help='If true, generate samples on the data, one full sweep at a time. Used to estimate raw performance without confidence_model')

    parser.add_argument('--use_complex_rmsd', action='store_true', 
                        default=False, 
                        help='If True, use complex RMSD values to train the confidence model.')

    parser.add_argument('--use_interface_rmsd', action='store_true', 
                        default=False, 
                        help='If True, use Interface RMSD values to train the confidence model.')

    parser.add_argument('--wandb_sweep', action='store_true', 
                        default=False, 
                        help='If True, use wandb sweep to optimize hyperparams for low-temp sampling')


    # iRMSD_pred

    # [contrastive]
    parser.add_argument("--contrastive", action="store_true", default=False,
                        help="If set, use additional contrastive learning loss")
    # [interface_reg]
    parser.add_argument("--interface_reg", action="store_true", default=False,
                        help="If set, use additional interface regularization loss")
                        

    args = parser.parse_args()
    process_args(args)
    return args


def process_args(args):
    """
        This function does a couple of nice things:
        1)  load any arguments specified in config_file
        2)  set default save_path and args_path if checkpoint provided
        3)  load any remaining arguments saved from checkpoint,
            if applicable. config_file takes precedence over saved
            args from checkpoint directory
    """

    # used for dispatcher only (bash script auto-formats to config)
    ## process run_name
    if args.run_name is None:
        args.run_name = args.save_path.split("/")[-1]

    # load configuration = override specified values
    ## load config_file
    if args.config_file is not None:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        override_args(args, config)

    # default checkpoint_path to save_path if mode == "test"
    #TODO: change back
    # if args.mode == "test":
    #     if args.checkpoint_path is None:
    #         args.checkpoint_path = args.save_path
    #     elif args.save_path == "":
    #         args.save_path = args.checkpoint_path

    # prepend output root (defaulted args.checkpoint_path to save_path)
    # the if and else are the same

    # check a path beginning with "model_best_" exists

    # Pattern to match
    pattern = os.path.join(args.checkpoint_path, 'fold_0/model_best_*')

    # Use glob to find matching paths
    matching_paths = glob.glob(pattern)

    if args.checkpoint_path is not None and matching_paths:
        args.args_file = os.path.join(args.checkpoint_path, args.args_file)
    else:
        args.args_file = os.path.join(args.save_path, args.args_file)
    args.log_file  = os.path.join(args.save_path, args.log_file)


    # data processing
    if args.data_process:
        return 

    # finally load all saved parameters
    if args.checkpoint_path is not None and matching_paths:
        if not os.path.exists(args.checkpoint_path):
            printt("invalid checkpoint_path", args.checkpoint_path)
        if os.path.exists(args.args_file):
            with open(args.args_file) as f:
                # saved_config = yaml.safe_load(f)
                # [self] yaml.load(..., Loader=yaml.Loader)
                saved_config = yaml.load(f, Loader=yaml.Loader)
        print("here",args.args_file)
        # do not overwrite certain args (inner keys and the below items, which are provided by the args)
        # outer_key is 'data', 'mode', ... inner_key is the correct key
        k_to_skip = [inner_key for outer_key in config.keys() for inner_key in config[outer_key].keys() ]
        # k_to_skip = list(config.keys())  # config takes precedence BUG!
        k_to_skip.extend(["checkpoint_path", "save_path",
                          "gpu", "mode", "test_fold",
                          "batch_size", "debug", "data_file", "num_gpu", "generate_n_predictions",
                          "samples_directory", "logger", "project", "config_file"])
        for k in k_to_skip:
            if k in saved_config:
                del saved_config[k]
        override_args(args, saved_config)


def override_args(args, config):
    """
        Recursively copy over config to args
    """
    # The override_args function you've provided is a Python function that recursively copies values from a configuration dictionary (config) to an object args. The function appears to be designed to take a configuration dictionary and update the attributes of an object (typically an instance of a class).
    for k,v in config.items():
        if type(v) is dict:
            override_args(args, v)
        else:
            args.__dict__[k] = v # args.__dict__: In Python, the __dict__ attribute of an object is a dictionary that contains the object's attributes and their values. It allows you to access and manipulate an object's attributes as if they were dictionary keys.


    return args

