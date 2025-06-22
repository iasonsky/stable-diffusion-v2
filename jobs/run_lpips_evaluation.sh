#!/bin/bash

echo "Launching parallel LPIPS evaluations..."

# Path to project root
ROOT_DIR="$(pwd)"
JOBS_DIR="${ROOT_DIR}/jobs"

# Create evaluation results directory
EVAL_DIR="${ROOT_DIR}/outputs/evaluations"
mkdir -p "$EVAL_DIR"

# Backbone to use
BACKBONE="alex"

# Path to the single job script
JOB_SCRIPT="${JOBS_DIR}/run_single_lpips_eval.sh"

echo "=== Submitting Parallel LPIPS Evaluation Jobs ==="
echo "Results will be saved to: $EVAL_DIR"
echo "Using backbone: $BACKBONE"
echo ""

# Job 1: Alpha blend with alpha=0.0
OUTPUT_FILE_1="${EVAL_DIR}/lpips_alpha_blend_0.0_${BACKBONE}.json"
JOB_NAME_1="lpips_alpha0.0"
echo "1. Submitting alpha_blend (α=0.0) evaluation..."
JOB_ID_1=$(sbatch --job-name="$JOB_NAME_1" --output="outputs/jobs/${JOB_NAME_1}_%A.out" --export=ALL,FUSION_TYPE="alpha_blend",ALPHA="0.0",OUTPUT_FILE="$OUTPUT_FILE_1",ROOT_DIR="$ROOT_DIR" "$JOB_SCRIPT" | awk '{print $4}')
echo "  Job ID: $JOB_ID_1"

# Job 2: Alpha blend with alpha=0.3
OUTPUT_FILE_2="${EVAL_DIR}/lpips_alpha_blend_0.3_${BACKBONE}.json"
JOB_NAME_2="lpips_alpha0.3"
echo "2. Submitting alpha_blend (α=0.3) evaluation..."
JOB_ID_2=$(sbatch --job-name="$JOB_NAME_2" --output="outputs/jobs/${JOB_NAME_2}_%A.out" --export=ALL,FUSION_TYPE="alpha_blend",ALPHA="0.3",OUTPUT_FILE="$OUTPUT_FILE_2",ROOT_DIR="$ROOT_DIR" "$JOB_SCRIPT" | awk '{print $4}')
echo "  Job ID: $JOB_ID_2"

# Job 3: Concat fusion
OUTPUT_FILE_3="${EVAL_DIR}/lpips_concat_${BACKBONE}.json"
JOB_NAME_3="lpips_concat"
echo "3. Submitting concat fusion evaluation..."
JOB_ID_3=$(sbatch --job-name="$JOB_NAME_3" --output="outputs/jobs/${JOB_NAME_3}_%A.out" --export=ALL,FUSION_TYPE="concat",ALPHA="none",OUTPUT_FILE="$OUTPUT_FILE_3",ROOT_DIR="$ROOT_DIR" "$JOB_SCRIPT" | awk '{print $4}')
echo "  Job ID: $JOB_ID_3"

echo ""
echo "=== All LPIPS Evaluation Jobs Submitted ==="
echo "Job IDs: $JOB_ID_1, $JOB_ID_2, $JOB_ID_3"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job logs:"
echo "  tail -f outputs/jobs/${JOB_NAME_1}_${JOB_ID_1}.out"
echo "  tail -f outputs/jobs/${JOB_NAME_2}_${JOB_ID_2}.out"
echo "  tail -f outputs/jobs/${JOB_NAME_3}_${JOB_ID_3}.out"
echo ""
echo "Results will be saved to:"
echo "  - $OUTPUT_FILE_1"
echo "  - $OUTPUT_FILE_2"
echo "  - $OUTPUT_FILE_3"
echo ""
echo "To check when all jobs are done:"
echo "  squeue -j $JOB_ID_1,$JOB_ID_2,$JOB_ID_3" 