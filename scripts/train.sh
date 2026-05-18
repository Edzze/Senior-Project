#!/usr/bin/env bash
set -euo pipefail

# --- Edit these paths for your machine ---
# If you run on WSL convert Windows paths to /mnt/c/... form, or run from Linux.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="${REPO_ROOT}/runs/wlasl_100_vitB16_32f_dec4x1024_FT"
CLIP_PATH="${REPO_ROOT}/CLIP_weights/ViT-L/ViT-L-14.pt"

# Replace with your dataset roots (example Windows path shown as comment)
# Windows example: C:\Users\man_t\OneDrive\Desktop\Senior Project\preprocessing\train
TRAIN_ROOT="/path/to/preprocessing/train"
VAL_ROOT="/path/to/preprocessing/test"

TRAIN_LIST="${REPO_ROOT}/WLASL_train100.txt"
VAL_LIST="${REPO_ROOT}/WLASL_test100.txt"
# ---------------------------------------

mkdir -p "${EXP_DIR}"

python -u -m torch.distributed.run --nproc_per_node=1 --master_port=25678 \
  main.py \
  --num_steps 10000 \
  --backbone "ViT-L/14-lnpre" \
  --backbone_type clip \
  --backbone_path "${CLIP_PATH}" \
  --decoder_num_layers 4 \
  --decoder_qkv_dim 1024 \
  --decoder_num_heads 16 \
  --num_classes 100 \
  --checkpoint_dir "${EXP_DIR}" \
  --auto_resume \
  --frames_available 1 \
  --train_data_root "${TRAIN_ROOT}" \
  --val_data_root "${VAL_ROOT}" \
  --train_list_path "${TRAIN_LIST}" \
  --val_list_path "${VAL_LIST}" \
  --n_shots -1 \
  --batch_size 16 \
  --batch_split 1 \
  --num_workers 16 \
  --num_frames 24 \
  --sampling_rate 4 \
  --num_spatial_views 3 \
  --num_temporal_views 1 \
  2>&1 | tee "${EXP_DIR}/train-$(date +%Y%m%d_%H%M%S).log"