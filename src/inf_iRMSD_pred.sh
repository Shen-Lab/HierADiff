#First Executable Line
# module load Anaconda3/2020.07
module load Anaconda3/2023.09-0
module load WebProxy/0000

module load GCC/11.3.0
module load CUDA/11.3.1

conda init bash
source ~/.bashrc
# conda activate diff_dock
conda activate pytorch_12




# LOCAL_WORLD_SIZE=2
# NNODES=1


# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}

# echo head node: $head_node

# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" /usr/bin/hostname --ip-address)
# # head_node_ip=$(srun --nodelist="$head_node" /usr/bin/hostname -I)

# echo Node IP: $head_node_ip

# CHECK='do while'
# while [[ ! -z $CHECK ]]; do
#     MPORT=$(( ( RANDOM % 60000 )  + 1025 ))
#     CHECK=$(sudo netstat -ap | grep $MPORT)
# done


export LOGLEVEL=INFO

# Caused by BUG in e3nn with NVFuser
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1


# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NUM_FOLDS=1  # number of seeds to try, default 5
SEED=803  # initial seed
# SEED=$1  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1 # this (args.gpu) will be OVERRIDE if in the DDP settings
NUM_GPU=1 # this (args.num_gpu) will be OVERRIDE if in the DDP settings
BATCH_SIZE=4  # for every gpu







### !!!!! ####
# --max_restarts 1 \




#### Load model with run name ####
# RUN_NAME="DEBUG_flexFM_Rt_2piRound"
# RUN_NAME="DEBUG_flexFM_Rt_secondLastLayerLigFlex_LargeModel" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_flexFM_Rt_secondLastLayerLigFlex_halfl1_LargeModel" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_flexFM_Rt_a100_indReadoutFlex_noSharedWeight_LargeModel" # 
# RUN_NAME="DEBUG_flexFM_Rt_a100_indReadoutFlex_sharedWeight_LargeModel" #

# RUN_NAME="DEBUG_flexFM_Rt_a100_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel"

# RUN_NAME="DEBUG_dips_flexFM_Rt_a100_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel"

# RUN_NAME="DEBUG_flexFM_Rt_a100_indReadoutFlex_noSharedWeight_resRotFlex_nearest_res_decay_LargeModel"

# RUN_NAME="DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight"

# flexSDE_Rt
# RUN_NAME="DEBUG_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_logScaling"
# RUN_NAME="DEBUG_db5_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_logScaling_test"

# RUN_NAME="DEBUG_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue"

# RUN_NAME="DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_ori_score"

# RUN_NAME="DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue"

# RUN_NAME="DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule"

# RUN_NAME="DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule_balanced_diffusion"

# RUN_NAME="DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule_balanced_diffusion_AffineEigenscale"

# 1-exp(-at)
# RUN_NAME="DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_20"
# RUN_NAME="DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule"
# RUN_NAME="DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_5"


#### Config file ####
# NAME="flexFM_Rt"
# NAME="flexFM_Rt_secondLastLayerLigFlex_LargeModel"  # change to name of config file
# NAME="flexFM_Rt_secondLastLayerLigFlex_halfl1_LargeModel"  # change to name of config file
# NAME="flexFM_Rt_indReadoutFlex_noSharedWeight_LargeModel"
# NAME="flexFM_Rt_indReadoutFlex_sharedWeight_LargeModel"

# NAME="flexFM_Rt_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel"

# NAME="dips_flexFM_Rt_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel"

# NAME="flexFM_Rt_indReadoutFlex_noSharedWeight_resRotFlex_nearest_res_decay_LargeModel"

# NAME="dips_flexFM_Rt_interModuleNoSharedWeight"

# flexSDE_Rt
# NAME="dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10"
# NAME="db5_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10"

# NAME="dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10"

# NAME="cont_dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10"

# NAME="cont_dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10_neg_drift"

# NAME="cont_dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10_squareSchedule"

# NAME="cont_dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10_squareSchedule_balanced_diffusion"

# NAME="cont_dips_flexSDE_Rt_interModuleNoSharedWeight_expSchedule_10_squareSchedule_balanced_diffusion_AffineEigenscale"

# 1-exp(-at)
# NAME="db5_ft_flexFM_Rt_interModuleNoSharedWeight_expSchedule_20"
# NAME="db5_ft_flexFM_Rt_interModuleNoSharedWeight_expSchedule"
# NAME="db5_ft_flexFM_Rt_interModuleNoSharedWeight_expSchedule_5"

#### Model path ####
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_2piRound/fold_0/model_best_47334_160_3.056_3.056.pth"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_secondLastLayerLigFlex_LargeModel/fold_0/model_every_115.pth" # 88, 122, 140 for val loss # 115 for both lig flex and val loss
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_secondLastLayerLigFlex_halfl1_LargeModel/fold_0/model_every_126.pth" # 86, 120, 126 for val loss
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_a100_indReadoutFlex_noSharedWeight_LargeModel/fold_0/model_every_108.pth" # 
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_a100_indReadoutFlex_sharedWeight_LargeModel/fold_0/model_every_120.pth" # 
# model_num=$1
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_a100_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel/fold_0/model_every_${model_num}.pth" #


# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_a100_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel/fold_0/model_every_100.pth" #

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_indReadoutFlex_sharedWeight_resRotFlex_randomRL_LargeModel/fold_0/model_last.pth" #

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_flexFM_Rt_a100_indReadoutFlex_noSharedWeight_resRotFlex_nearest_res_decay_LargeModel/fold_0/model_every_33.pth" #


# model_num=$1
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight/fold_0/model_every_${model_num}.pth" #
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight/fold_0/model_every_68.pth" # train on dips: before epoch 73; finetune on DB5.5: epoch 85/88

# flexSDE_Rt
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_logScaling/fold_0/model_every_117.pth" 
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_db5_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_logScaling_test/fold_0/model_every_100.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue/fold_0/model_last.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_ori_score/fold_0/model_last.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue/fold_0/model_last.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule/fold_0/model_last.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule_balanced_diffusion/fold_0/model_every_20.pth"

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_cont_dips_flexSDE_Rt_a100_interModuleNoSharedWeight_expSchedule_10_scaledEigenValue_squareSchedule_balanced_diffusion_AffineEigenscale/fold_0/model_every_47.pth"


# 1-exp(-at)
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_20/fold_0/model_every_124.pth"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule/fold_0/model_every_146.pth"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_5/fold_0/model_every_165.pth"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_5/fold_0/model_every_116.pth"


# NAME="db5_iRMSD_pred"  # change to name of config file
# RUN_NAME="DEBUG_db5_iRMSD_pred" # should uniauely describe the current experiment
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_db5_iRMSD_pred/fold_0/model_best_60_11_1.986_1.986.pth"

NAME="db5_iRMSD_pred_contr"  # change to name of config file
RUN_NAME="DEBUG_db5_iRMSD_pred_contr" # should uniauely describe the current experiment
MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_db5_iRMSD_pred_contr/fold_0/model_best_12075_1002_0.993_0.993.pth"

# model_num=$1
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_5/fold_0/model_every_${model_num}.pth"

export SAMPLING_MODEL_PATH=${MODEL_PATH}

#### LDDT_OUT control####
LDDT_OUT="True"
export LDDT_OUT=${LDDT_OUT}
PICKLE_INDEX=$1
export PICKLE_INDEX=${PICKLE_INDEX}



# #### Other Dataset (optional) ####
# # data_file: 
# DATA_FILE="datasets/DB5/splits_medDiff.csv"
# DATA_FILE="datasets/DIPS/data_file_cov_80.csv"
# DATA_FILE="datasets/DIPS/hetero_data.csv"
DATA_FILE="datasets/DB5/splits.csv"
# # data_path: 
DATA_PATH="datasets/DB5/pairs_pruned"
# DATA_PATH="datasets/DIPS/pairs_pruned"
export DATA_FILE=${DATA_FILE}
export DATA_PATH=${DATA_PATH}

#### SPLIT ####
SPLIT="val"  #"train"   #"test"
export SPLIT=${SPLIT}

#### OUTPUT_PATH #### "eva_out/{sampling log} and {result_pkl} and ${RUN_NAME}_${SPLIT}/epoch-{}/{pdbs} and {stat_pkl}
OUTPUT_PATH="eva_out/${RUN_NAME}_${SPLIT}"
export OUTPUT_PATH=${OUTPUT_PATH}

#### visulaization number ####
visualize_n_val_graphs=30
export visualize_n_val_graphs=${visualize_n_val_graphs}

#### seed ####
export SEED=${SEED}






CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"

echo SAVE_PATH: $SAVE_PATH

python src/main_inf_iRMSD_pred_1gpu.py \
--mode "train" \
--config_file $CONFIG \
--run_name $RUN_NAME \
--save_path $SAVE_PATH \
--batch_size $BATCH_SIZE \
--num_folds $NUM_FOLDS \
--num_gpu $NUM_GPU \
--gpu $CUDA --seed $SEED \
--logger "wandb" \
--project "DynamicBindFM tuning" \
--visualize_n_val_graphs 0 \
--visualization_path $VISUALIZATION_PATH \
--checkpoint_path $SAVE_PATH \
--sampling_model_pth $MODEL_PATH \