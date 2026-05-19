#!/usr/bin/env bash
set -euo pipefail

# WONN CIFAR script: ch=64 -> ch_final=256.
# Place this file in the same directory as train.py, e.g. image_recognition/cifar/.
#
# Default setting: CIFAR-10 + attentive coupling + learned MLP S/I functions.
# To run other settings, override environment variables, for example:
#   DATA=cifar100 DATA_ROOT=/path/to/data bash run_cifar_ch64to256.sh
#   COUPLING=conv SI_FUNC=trig DATA_ROOT=/path/to/data bash run_cifar_ch64to256.sh
#
# Important:
#   - DATA_ROOT must point to the directory containing CIFAR data.
#   - COUPLING can be: attn or conv.
#   - SI_FUNC can be: mlp or trig.
#   - Do not remove --ch_final 256; this is the official ch64->256 setting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA=${DATA:-cifar10}
DATA_ROOT=${DATA_ROOT:-./data}
COUPLING=${COUPLING:-attn}
SI_FUNC=${SI_FUNC:-mlp}

GPUS=${GPUS:-0}
NPROC=${NPROC:-1}

EPOCHS=${EPOCHS:-200}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}
EVAL_FREQ=${EVAL_FREQ:-1}
BS=${BS:-64}
EVAL_BS=${EVAL_BS:-64}
WORKERS=${WORKERS:-8}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
BETA=${BETA:-0.99}
GRAD_CLIP=${GRAD_CLIP:-0.0}
SEED=${SEED:-137}

EXP_NAME=${EXP_NAME:-WONN_${DATA}_${COUPLING}_${SI_FUNC}_ch64to256}

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCH=(python)
fi

"${LAUNCH[@]}" image_recognition/cifar/train.py "${EXP_NAME}" \
  --data "${DATA}" \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --eval_freq "${EVAL_FREQ}" \
  --batchsize "${BS}" \
  --eval_batchsize "${EVAL_BS}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta "${BETA}" \
  --grad_clip "${GRAD_CLIP}" \
  --L 6 \
  --T 3 \
  --ch 64 \
  --ch_final 256 \
  --gamma 0.1 \
  --coupling "${COUPLING}" \
  --si_func "${SI_FUNC}" \
  --kernel_sizes 7 5 5 3 3 3 \
  --group_size 2 \
  --hidden_ratio 2 \
  --input_patch_size 4 \
  --output_ksize 3 \
  --norm gn \
  --seed "${SEED}" \
  --amp False \
  --amp_dtype bf16 \
  --compile True \
  --compile_mode default \
  --compile_backend inductor \
  --compile_dynamic False
