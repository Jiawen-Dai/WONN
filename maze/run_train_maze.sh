#!/usr/bin/env bash
set -euo pipefail

# WONN Maze-Hard script: Setting from the  Winfree maze run.
# Place this file in the maze/ directory.
#
# Default setting:
#   L=1, T=24, group_size=1, ch=128
#   lr=1e-3, beta=0.995, seed=137
#   free/path class weights = 5.0
#
# Example:
#   GPUS=0 bash maze/run_train_maze.sh
#   GPUS=0,1,2,3 NPROC=4 bash maze/run_train_maze.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-}

if [ -z "${DATA_ROOT}" ]; then
  echo "Error: DATA_ROOT is required."
  echo "Example:"
  echo "  DATA_ROOT=/path/to/maze-30x30-hard-1k GPUS=0 bash maze/run_train_maze.sh"
  exit 1
fi

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}

EPOCHS=${EPOCHS:-9000}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-100}
EVAL_FREQ=${EVAL_FREQ:-10}
BS=${BS:-64}
WORKERS=${WORKERS:-4}
LR=${LR:-1e-3}
GRAD_CLIP=${GRAD_CLIP:-0.0}
BETA=${BETA:-0.995}
SEED=${SEED:-137}

L=${L:-1}
T=${T:-24}
CH=${CH:-128}
HEADS=${HEADS:-8}
GROUP_SIZE=${GROUP_SIZE:-1}
GAMMA=${GAMMA:-0.1}
NORM=${NORM:-gn}
COUPLING=${COUPLING:-attn}
OUTPUT_KSIZE=${OUTPUT_KSIZE:-3}

W_WALL=${W_WALL:-1.0}
W_FREE=${W_FREE:-5.0}
W_START=${W_START:-2.0}
W_GOAL=${W_GOAL:-2.0}
W_PATH=${W_PATH:-5.0}

AMP=${AMP:-True}
AMP_DTYPE=${AMP_DTYPE:-bf16}
COMPILE=${COMPILE:-False}
COMPILE_MODE=${COMPILE_MODE:-default}
COMPILE_BACKEND=${COMPILE_BACKEND:-inductor}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-False}

EXP_NAME=${EXP_NAME:-maze_L${L}T${T}_g${GROUP_SIZE}_ch${CH}_e${EPOCHS}_lr1e3_b0995_s${SEED}}

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCH=(python)
fi

"${LAUNCH[@]}" maze/train.py \
  --exp_name "${EXP_NAME}" \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --eval_freq "${EVAL_FREQ}" \
  --batchsize "${BS}" \
  --num_workers "${WORKERS}" \
  --lr "${LR}" \
  --grad_clip "${GRAD_CLIP}" \
  --beta "${BETA}" \
  --seed "${SEED}" \
  --L "${L}" \
  --T "${T}" \
  --ch "${CH}" \
  --heads "${HEADS}" \
  --group_size "${GROUP_SIZE}" \
  --gamma "${GAMMA}" \
  --norm "${NORM}" \
  --coupling "${COUPLING}" \
  --output_ksize "${OUTPUT_KSIZE}" \
  --w_wall "${W_WALL}" \
  --w_free "${W_FREE}" \
  --w_start "${W_START}" \
  --w_goal "${W_GOAL}" \
  --w_path "${W_PATH}" \
  --amp "${AMP}" \
  --amp_dtype "${AMP_DTYPE}" \
  --compile "${COMPILE}" \
  --compile_mode "${COMPILE_MODE}" \
  --compile_backend "${COMPILE_BACKEND}" \
  --compile_dynamic "${COMPILE_DYNAMIC}"