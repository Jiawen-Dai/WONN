#!/usr/bin/env bash
set -euo pipefail

# WONN ImageNet-100 script: ch=128, constant channels.
# Place this file in image_recognition/imagenet/.
#
# Default setting: ImageNet-100 + attentive coupling + learned MLP S/I functions.
# To run other settings, override environment variables, for example:
#   DATA_ROOT=/path/to/imagenet-100 bash run_imagenet100_ch128.sh
#   COUPLING=conv SI_FUNC=trig DATA_ROOT=/path/to/imagenet-100 bash run_imagenet100_ch128.sh
#
# Important:
#   - DATA_ROOT must point to an ImageFolder-style ImageNet-100 root containing train/ and val/.
#   - COUPLING can be: attn or conv.
#   - SI_FUNC can be: mlp or trig.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-}
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

EXP_NAME=${EXP_NAME:-WONN_imagenet100_${COUPLING}_${SI_FUNC}_ch128}

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCH=(python)
fi

"${LAUNCH[@]}" image_recognition/imagenet/train_imagenet100.py "${EXP_NAME}" \
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
  --ch 128 \
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
