#First Executable Line
# module load Anaconda3/2020.07
module load Anaconda3/2023.09-0
module load WebProxy

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
BATCH_SIZE=8  # for every gpu


if [ -z "$2" ] || [ -z "$3" ]; then
    echo "No subset sampling"
else
    echo "Subset sampling"
    export sub_ind=$2
    export total_ind=$3
fi







### !!!!! ####
# --max_restarts 1 \




#### Load model with run name ####
# RUN_NAME="DEBUG_confidence_db5_finalSteps" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_confidence_db5_finalSteps_wirmsd" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_confidence_db5_val" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_confidence_db5_val_wirmsd" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3" # should uniauely describe the current experiment
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.2" # should uniauely describe the current experiment


# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm_1e-4" # should uniauely describe the current experiment
# xxx #RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm" # should uniauely describe the current experiment




#### Config file ####
# NAME="confidence_db5_finalSteps"  # change to name of config file
# NAME="confidence_db5_finalSteps_wirmsd"  # change to name of config file
# NAME="confidence_db5_val"  # change to name of config file
# NAME="confidence_db5_val_wirmsd"  # change to name of config file
# NAME="confidence_db5_finalSteps_dropout0.3"  # change to name of config file
# NAME="confidence_db5_finalSteps_dropout0.2"  # change to name of config file

# NAME="confidence_db5_finalSteps_dropout0.3_l1norm_1e-4"  # change to name of config file
# xxxx # NAME="confidence_db5_finalSteps_dropout0.3_l1norm"  # change to name of config file


#### Model path ####
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps/fold_0/model_every_202.pth" # 188 from best model and 189,202!! from wandb
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_wirmsd/fold_0/model_every_194.pth" # 194!!, 256 from wandb # 190 from best model
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_val/fold_0/model_every_202.pth" # 193, 202 from wandb # 192 from best model
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_val_wirmsd/fold_0/model_every_205.pth" # 205, 232 from wandb # 197 from best model
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3/fold_0/model_every_227.pth" # 190 from best model and 198!!,227 from wandb
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.2/fold_0/model_every_235.pth" # 235 from wandb # 187 from best model

# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm_1e-4/fold_0/model_every_234.pth" # 202, 234 from wandb # 201 from best model
#xxx # MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm/fold_0/model_every_212.pth" # 212 from wandb # 201 from best model xxxxx

#
# dropout0.3_l1norm_1e-6
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm_1e-6" # should uniauely describe the current experiment
# NAME="confidence_db5_finalSteps_dropout0.3_l1norm_1e-6"  # change to name of config file
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l1norm_1e-6/fold_0/model_every_253.pth" # 192, 216, 218, 253, 290 from wandb # 196, 198, 201, 204, 215, 217,219, 221, 227, 257 from best model

# finalSteps_wirmsd_split0
# NAME="confidence_db5_finalSteps_wirmsd_split0"  # change to name of config file
# RUN_NAME="DEBUG_confidence_db5_finalSteps_wirmsd_split0" # should uniauely describe the current experiment
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_wirmsd_split0/fold_0/model_every_$2.pth" # 200 to 207 from wandb # 200 from best model

# # ens0
# NAME="confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens0"
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens0"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens0/fold_0/model_every_$2.pth" # 194-210 from wandb 

# # ens3
# NAME="confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens3"
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens3"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens3/fold_0/model_every_$2.pth" # 203 234 from wandb

# # ens4
# NAME="confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens4"
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens4"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens4/fold_0/model_every_$2.pth" # 200 213 from wandb

# # ens6
# NAME="confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens5"
# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens5"
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5_ens5/fold_0/model_every_$2.pth" # 194 214 241 from wandb


# NAME="confidence_coldStart_db5_finalSteps_dropout0.3"  # change to name of config file
# RUN_NAME="DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3" # should uniauely describe the current experiment
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3/fold_0/model_every_$2.pth" # 194-210 from wandb

# NAME="confidence_coldStart_db5_finalSteps_dropout0.3_l2norm_1e-5"  # change to name of config file
# RUN_NAME="DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3_l2norm_1e-5" # should uniauely describe the current experiment
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3_l2norm_1e-5/fold_0/model_every_51.pth" #model_best_13208_51_0.034_0.034.pth"

NAME="confidence_coldStart_db5_finalSteps_dropout0.3"  # change to name of config file
RUN_NAME="DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3" # should uniauely describe the current experiment
MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_coldStart_db5_finalSteps_dropout0.3/fold_0/model_every_51.pth" #model_best_13208_51_0.034_0.034.pth"

DATA_FILE="datasets/DB5/splits.csv"
# # data_path: 
DATA_PATH="datasets/DB5/pairs_pruned"
# DATA_PATH="datasets/DIPS/pairs_pruned"



# RUN_NAME="DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5" # should uniauely describe the current experiment
# NAME="confidence_db5_finalSteps_dropout0.3_l2norm_1e-5"  # change to name of config file
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_confidence_db5_finalSteps_dropout0.3_l2norm_1e-5/fold_0/model_every_212.pth" # 212, 217, 225, 240, 249 from wandb # 248 from best model

# # data_file:
# DATA_FILE="datasets/DB5_samples/sample.csv"
# # data_path:
# DATA_PATH="datasets/DB5_samples/pairs_pruned"



# model_num=$1
# MODEL_PATH="/scratch/user/rujieyin/DyDock/ckpts/DEBUG_dips_flexFM_Rt_a100_interModuleNoSharedWeight_expSchedule_5/fold_0/model_every_${model_num}.pth"

export SAMPLING_MODEL_PATH=${MODEL_PATH}

#### LDDT_OUT control####
LDDT_OUT="True"
export LDDT_OUT=${LDDT_OUT}
PICKLE_INDEX=$1
export PICKLE_INDEX=${PICKLE_INDEX}

#### subset sampling ####
# if $2 and $3 are not empty, then subset the sampling






# #### Other Dataset (optional) ####
# # data_file: 
# DATA_FILE="datasets/DB5/splits_medDiff.csv"
# DATA_FILE="datasets/DIPS/data_file_cov_80.csv"
# DATA_FILE="datasets/DIPS/hetero_data.csv"



export DATA_FILE=${DATA_FILE}
export DATA_PATH=${DATA_PATH}

#### SPLIT ####
SPLIT="test"  #"train" #"test" # 
BATCH_SIZE=1
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

python src/main_inf_conf_1gpu.py \
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
