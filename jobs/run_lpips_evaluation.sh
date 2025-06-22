#!/bin/bash

echo "Launching parallel LPIPS evaluations..."

# Path to project root
ROOT_DIR="$(pwd)"
JOBS_DIR="${ROOT_DIR}/jobs"

# Create evaluation results directory
EVAL_DIR="${ROOT_DIR}/outputs/evaluations"
mkdir -p "$EVAL_DIR"

# Define backbones to evaluate - you can easily modify this list
BACKBONES=("alex" "vgg" "squeeze")

# Define evaluation methods - you can easily modify this list
METHODS=(
  "alpha_blend:0.0"
  "alpha_blend:0.3"
  "concat:none"
)

# Path to the single job script
JOB_SCRIPT="${JOBS_DIR}/run_single_lpips_eval.sh"

echo "=== Submitting Parallel LPIPS Evaluation Jobs ==="
echo "Results will be saved to: $EVAL_DIR"
echo "Backbones to evaluate: ${#BACKBONES[@]} (${BACKBONES[*]})"
echo "Methods to evaluate: ${#METHODS[@]}"
for method in "${METHODS[@]}"; do
  echo "  - $method"
done
echo "Total jobs to submit: $((${#BACKBONES[@]} * ${#METHODS[@]}))"
echo ""

# Submit jobs for each backbone and method combination
JOB_IDS=()
JOB_NAMES=()
OUTPUT_FILES=()
SHARED_FILES=()

job_counter=1
for BACKBONE in "${BACKBONES[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    FUSION_TYPE=$(echo "$METHOD" | cut -d':' -f1)
    ALPHA=$(echo "$METHOD" | cut -d':' -f2)
    
    # Create descriptive names
    if [ "$ALPHA" = "none" ]; then
      JOB_NAME="lpips_${FUSION_TYPE}_${BACKBONE}"
      OUTPUT_FILE="${EVAL_DIR}/lpips_${FUSION_TYPE}_${BACKBONE}.json"
    else
      JOB_NAME="lpips_${FUSION_TYPE}_${ALPHA}_${BACKBONE}"
      OUTPUT_FILE="${EVAL_DIR}/lpips_${FUSION_TYPE}_${ALPHA}_${BACKBONE}.json"
    fi
    
    echo "${job_counter}. Submitting ${FUSION_TYPE} (α=${ALPHA}) with ${BACKBONE} backbone..."
    JOB_ID=$(sbatch \
      --job-name="$JOB_NAME" \
      --output="outputs/jobs/${JOB_NAME}_%A.out" \
      --export=ALL,FUSION_TYPE="$FUSION_TYPE",ALPHA="$ALPHA",OUTPUT_FILE="$OUTPUT_FILE",ROOT_DIR="$ROOT_DIR",BACKBONE="$BACKBONE" \
      "$JOB_SCRIPT" | awk '{print $4}')
    
    echo "  Job ID: $JOB_ID"
    
    JOB_IDS+=("$JOB_ID")
    JOB_NAMES+=("$JOB_NAME")
    OUTPUT_FILES+=("$OUTPUT_FILE")
    
    job_counter=$((job_counter + 1))
  done
  
  # Track shared summary files
  SHARED_FILES+=("${EVAL_DIR}/lpips_summary_${BACKBONE}.json")
done

echo ""
echo "=== All LPIPS Evaluation Jobs Submitted ==="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job logs:"
for i in "${!JOB_NAMES[@]}"; do
  echo "  tail -f outputs/jobs/${JOB_NAMES[$i]}_${JOB_IDS[$i]}.out"
done
echo ""
echo "Individual results will be saved to:"
for output_file in "${OUTPUT_FILES[@]}"; do
  echo "  - $output_file"
done
echo ""
echo "Shared summary files will be saved to:"
for shared_file in "${SHARED_FILES[@]}"; do
  echo "  - $shared_file"
done
echo ""
echo "To check when all jobs are done:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")" 