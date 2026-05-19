#!/usr/bin/env bash
set -euo pipefail

# WONN ImageNet-1K script: ch=256, constant channels.
# Place this file in image_recognition/imagenet/.
#
# Default setting: 4-GPU DDP + attentive coupling + learned MLP S/I functions.
# To run other settings, override environment variables, for example:
#   GPUS=4,5,6,7 NPROC=4 DATA_ROOT=/data/imagenet_common bash run_imagenet1k_ch256.sh
#   COUPLING=conv SI_FUNC=trig bash run_imagenet1k_ch256.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:-/data/imagenet_common}
COUPLING=${COUPLING:-attn}
SI_FUNC=${SI_FUNC:-mlp}

GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}

EPOCHS=${EPOCHS:-300}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-20}
EVAL_FREQ=${EVAL_FREQ:-2}
BS=${BS:-128}
EVAL_BS=${EVAL_BS:-128}
WORKERS=${WORKERS:-8}
LR=${LR:-7.5e-4}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
MIN_LR=${MIN_LR:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
BETA=${BETA:-0.99}
GRAD_CLIP=${GRAD_CLIP:-0.0}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}
CUTMIX_ALPHA=${CUTMIX_ALPHA:-1.0}
CUTMIX_PROB=${CUTMIX_PROB:-0.8}
SEED=${SEED:-137}

EXP_NAME=${EXP_NAME:-WONN_imagenet1k_${COUPLING}_${SI_FUNC}_ch256}

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ "${NPROC}" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCH=(python)
fi

"${LAUNCH[@]}" image_recognition/imagenet/train_imagenet1k.py "${EXP_NAME}" \
  --data_root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --eval_freq "${EVAL_FREQ}" \
  --batchsize "${BS}" \
  --eval_batchsize "${EVAL_BS}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --min_lr "${MIN_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta "${BETA}" \
  --grad_clip "${GRAD_CLIP}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --cutmix_alpha "${CUTMIX_ALPHA}" \
  --cutmix_prob "${CUTMIX_PROB}" \
  --L 6 \
  --T 3 \
  --ch 256 \
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
  --amp True \
  --amp_dtype bf16 \
  --compile True \
  --compile_mode default \
  --compile_backend inductor \
  --compile_dynamic False
