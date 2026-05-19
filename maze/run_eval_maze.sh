#!/usr/bin/env bash
set -euo pipefail

# WONN Maze-Hard evaluation script.
#
# Default setting:
#   K = 32 random initializations per board
#   T_eval = 25
#   energy voting by both final energy and path-sum energy
#
# Example:
#   DATA_ROOT=/path/to/maze-30x30-hard-1k GPUS=0 bash maze/run_eval_maze.sh
#   DATA_ROOT=/path/to/maze-30x30-hard-1k MODEL_PATH=runs/maze/YOUR_RUN/ema_model.pth GPUS=0 bash maze/run_eval_maze.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-}

if [ -z "${DATA_ROOT}" ]; then
  echo "Error: DATA_ROOT is required."
  echo "Example:"
  echo "  DATA_ROOT=/path/to/maze-30x30-hard-1k GPUS=0 bash maze/run_eval_maze.sh"
  exit 1
fi

SPLIT=${SPLIT:-test}

TRAIN_EXP_NAME=${TRAIN_EXP_NAME:-maze_L1T24_g1_ch128_e9000_lr1e3_b0995_s137}
MODEL_PATH=${MODEL_PATH:-runs/maze/${TRAIN_EXP_NAME}/ema_model.pth}

GPUS=${GPUS:-0}

BS=${BS:-32}
WORKERS=${WORKERS:-4}

L=${L:-1}
T=${T:-25}
CH=${CH:-128}
HEADS=${HEADS:-8}
GROUP_SIZE=${GROUP_SIZE:-1}
GAMMA=${GAMMA:-0.1}
NORM=${NORM:-gn}
COUPLING=${COUPLING:-attn}
OUTPUT_KSIZE=${OUTPUT_KSIZE:-3}

NUM_INITS=${NUM_INITS:-32}
VOTE_MODE=${VOTE_MODE:-both}
ENERGY_LAYER=${ENERGY_LAYER:--1}

SEED=${SEED:-137}

export CUDA_VISIBLE_DEVICES="${GPUS}"

CMD=(python maze/eval_maze.py
  --model_path "${MODEL_PATH}"
  --data_root "${DATA_ROOT}"
  --split "${SPLIT}"
  --batchsize "${BS}"
  --num_workers "${WORKERS}"
  --L "${L}"
  --T "${T}"
  --ch "${CH}"
  --heads "${HEADS}"
  --gamma "${GAMMA}"
  --group_size "${GROUP_SIZE}"
  --norm "${NORM}"
  --coupling "${COUPLING}"
  --output_ksize "${OUTPUT_KSIZE}"
  --num_inits "${NUM_INITS}"
  --vote_mode "${VOTE_MODE}"
  --energy_layer "${ENERGY_LAYER}"
  --seed "${SEED}"
)

"${CMD[@]}"