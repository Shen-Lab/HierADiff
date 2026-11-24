#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/infer_unbound.sh INPUT_DIR [SEED] [NUM_STEPS] [OUTPUT_DIR]

INPUT_DIR="${1:?Provide a directory containing receptor.pdb and ligand.pdb}"
SEED="${2:-7}"
NUM_STEPS="${3:-40}"
OUTPUT_DIR="${4:-}"

cd "$(dirname "$0")/.."
ROOT="$PWD"

MODEL_DIR="ckpts/DEBUG_db5_ft_flexFM_Rt_interSWReadoutSW_adaptive_schedule_fape_bothNMFeat_readOutInterAtt"
CONFIG="config/db5_flexFM_Rt_interSWReadoutSW_adaptive_schedule_fape_bothNMFeat_readOutInterAtt.yaml"

export DATA_PATH="$(realpath "$INPUT_DIR")"
export SAMPLING_MODEL_PATH="$ROOT/$MODEL_DIR/fold_0/model_every_141.pth"
export SEED
export SPLIT=test

if [[ -n "$OUTPUT_DIR" ]]; then
    export OUTPUT_PATH="$(realpath -m "$OUTPUT_DIR")"
else
    unset OUTPUT_PATH
fi

python src/main_inf_nm_1gpu.py \
    --config_file "$CONFIG" \
    --checkpoint_path "$MODEL_DIR" \
    --mode test \
    --num_steps "$NUM_STEPS"
