#!/usr/bin/env bash
set -euo pipefail

# WONN Sudoku script.
# Place this file in the sudoku/ directory.
#
# Default setting follows the old Winfree Sudoku run:
#   L=1, T=16, group_size=1, ch=256
#   lr=1e-3, beta=0.995, seed=137
#   amp=bf16, compile=True
#
# Example:
#   DATA_ROOT=/path/to/sudoku GPUS=0 bash sudoku/run_train_sudoku.sh
#   DATA_ROOT=/path/to/sudoku GPUS=0,1,2,3 NPROC=4 bash sudoku/run_train_sudoku.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-}

if [ -z "${DATA_ROOT}" ]; then
  echo "Error: DATA_ROOT is required."
  echo "Example:"
  echo "  DATA_ROOT=/path/to/sudoku GPUS=0 bash sudoku/run_train_sudoku.sh"
  exit 1
fi

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}

EPOCHS=${EPOCHS:-100}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-20}
EVAL_FREQ=${EVAL_FREQ:-1}
BS=${BS:-100}
EVAL_BS=${EVAL_BS:-100}
WORKERS=${WORKERS:-8}
LR=${LR:-1e-3}
GRAD_CLIP=${GRAD_CLIP:-0.0}
BETA=${BETA:-0.995}
SEED=${SEED:-137}

L=${L:-1}
T=${T:-16}
CH=${CH:-256}
HEADS=${HEADS:-8}
GROUP_SIZE=${GROUP_SIZE:-1}
GAMMA=${GAMMA:-0.1}
NORM=${NORM:-gn}
COUPLING=${COUPLING:-attn}
OUTPUT_KSIZE=${OUTPUT_KSIZE:-3}

AMP=${AMP:-True}
AMP_DTYPE=${AMP_DTYPE:-bf16}
COMPILE=${COMPILE:-True}
COMPILE_MODE=${COMPILE_MODE:-default}
COMPILE_BACKEND=${COMPILE_BACKEND:-inductor}
COMPILE_DYNAMIC=${COMPILE_DYNAMIC:-False}

EXP_NAME=${EXP_NAME:-sudoku_id_L${L}T${T}G${GROUP_SIZE}_csin_ch${CH}_seed${SEED}}

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCH=(python)
fi

"${LAUNCH[@]}" sudoku/train.py \
  --exp_name "${EXP_NAME}" \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --eval_freq "${EVAL_FREQ}" \
  --batchsize "${BS}" \
  --eval_batchsize "${EVAL_BS}" \
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
  --amp "${AMP}" \
  --amp_dtype "${AMP_DTYPE}" \
  --compile "${COMPILE}" \
  --compile_mode "${COMPILE_MODE}" \
  --compile_backend "${COMPILE_BACKEND}" \
  --compile_dynamic "${COMPILE_DYNAMIC}"