#!/bin/bash

ROOT_DIR="/home/scur2690/stable-diffusion-v2"

# Throttling parameters
MAX_JOBS=3
SLEEP_TIME=60

sanitize() {
  echo "$1" | tr '/: ' '__'
}

wait_for_available_slot() {
  while true; do
    CURRENT_JOBS=$(squeue -u "$USER" -h | wc -l)
    if (( CURRENT_JOBS < MAX_JOBS )); then
      break
    fi
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] ⏳ Too many jobs queued ($CURRENT_JOBS). Waiting for available slot..."
    sleep "$SLEEP_TIME"
  done
}

# Alphas and model paths
ALPHAS=(0.1 0.2 0.3)
ALIGNER_MODELS=(
  "/scratch-shared/holy-triangle/weights/img2text_aligner_fixed/flickr30k_infonce/model_best.pth"
)

PROMPT="a photo of a cat"
REF_IMG="${ROOT_DIR}/data/van_gogh_starry_night.jpg"
CONFIG="${ROOT_DIR}/configs/stable-diffusion/v2-inference-v.yaml"
CKPT="/scratch-shared/holy-triangle/weights/stable-diffusion-2-1/v2-1_768-ema-pruned.ckpt"

JOB_SCRIPT="${ROOT_DIR}/jobs/run_single_alpha.sh"

for ALIGNER_MODEL in "${ALIGNER_MODELS[@]}"; do
  SAFE_MODEL_NAME=$(sanitize "$ALIGNER_MODEL")
  SAFE_PROMPT=$(sanitize "$PROMPT")

  for ALPHA in "${ALPHAS[@]}"; do
    wait_for_available_slot
    sleep 2

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] 📤 Submitting PNO job: alpha=$ALPHA"

    sbatch \
      --job-name="pno_a${ALPHA}" \
      --output="${ROOT_DIR}/outputs/jobs/pno_a${ALPHA}_p${SAFE_PROMPT}_%A.out" \
      --partition=gpu_a100 \
      --gpus=1 \
      --cpus-per-task=16 \
      --time=00:05:00 \
      --mem=40G \
      --export=ALL,ALPHA=$ALPHA,PROMPT="$PROMPT",REF_IMG="$REF_IMG",ALIGNER_MODEL="$ALIGNER_MODEL",CONFIG="$CONFIG",CKPT="$CKPT",ROOT_DIR="$ROOT_DIR" \
      "$JOB_SCRIPT"
  done
done
