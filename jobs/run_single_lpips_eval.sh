#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=20G

echo "Starting LPIPS evaluation for: fusion_type=${FUSION_TYPE}, alpha=${ALPHA}"
echo "Job ID: $SLURM_JOB_ID"

# Path to project root
ROOT_DIR="$(pwd)"
cd "$ROOT_DIR"

# When using UV on a100 - fix VIRTUAL_ENV warning
source .venv/bin/activate
unset VIRTUAL_ENV

# Create evaluation results directory
EVAL_DIR="${ROOT_DIR}/outputs/evaluations"
mkdir -p "$EVAL_DIR"

# Base path for experiments
BASE_PATH="${ROOT_DIR}/outputs/txt2img-samples"

# Backbone to use
BACKBONE="vgg"

echo "=== Running LPIPS Evaluation ==="
echo "Base path: $BASE_PATH"
echo "Fusion type: $FUSION_TYPE"
echo "Alpha: $ALPHA"
echo "Backbone: $BACKBONE"
echo "Output file: $OUTPUT_FILE"
echo ""

# Build the command based on whether alpha is specified
if [ "$ALPHA" = "none" ]; then
    # For concat fusion (no alpha parameter)
    CMD="uv run python scripts/quantitative_metrics/compute_lpips_score.py \
      --base_path \"$BASE_PATH\" \
      --fusion_type \"$FUSION_TYPE\" \
      --backbone \"$BACKBONE\" \
      --output_file \"$OUTPUT_FILE\""
else
    # For alpha_blend fusion (with alpha parameter)
    CMD="uv run python scripts/quantitative_metrics/compute_lpips_score.py \
      --base_path \"$BASE_PATH\" \
      --fusion_type \"$FUSION_TYPE\" \
      --alpha \"$ALPHA\" \
      --backbone \"$BACKBONE\" \
      --output_file \"$OUTPUT_FILE\""
fi

echo "Running: $CMD"
eval $CMD

# Check if evaluation was successful
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "=== Evaluation Complete ==="
    echo "Results saved to: $OUTPUT_FILE"
    
    # Print quick summary
    SUMMARY=$(python -c "
import json
try:
    with open('$OUTPUT_FILE') as f:
        data = json.load(f)
    summary = data['summary']
    print(f\"Processed {summary['total_samples']} samples from {summary['total_folders']} folders\")
    if summary['overall_avg_lpips'] is not None:
        print(f\"Average LPIPS: {summary['overall_avg_lpips']:.4f} (±{summary['overall_std_lpips']:.4f})\")
        print(f\"Min LPIPS: {summary['overall_min_lpips']:.4f}, Max LPIPS: {summary['overall_max_lpips']:.4f}\")
    else:
        print('No valid results found')
except Exception as e:
    print(f'Error reading results: {e}')
")
    
    echo "$SUMMARY"
else
    echo "ERROR: Evaluation failed - output file not created"
    exit 1
fi

echo ""
echo "LPIPS evaluation job completed successfully!" 