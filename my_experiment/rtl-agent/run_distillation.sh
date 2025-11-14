#!/bin/bash
# On-Policy Distillation Training Script for RTL Code Generation
# Distills a large model (teacher) into a smaller model (student)

set -e  # Exit on error

echo "====================================================================="
echo "RTL Agent - On-Policy Distillation Training"
echo "====================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"

# Default dataset path - you can override this
DATASET_PATH="${1:-cvdp_5_problems.jsonl}"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Warning: Dataset not found at $DATASET_PATH"
    echo "You can create it by selecting 5 problems from the full dataset:"
    echo "  head -n 5 /home/ubuntu/peter/benchmark/cvdp_benchmark/example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl > cvdp_5_problems.jsonl"
    echo ""
    read -p "Do you want to use the full example dataset instead? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DATASET_PATH="/home/ubuntu/peter/benchmark/cvdp_benchmark/example_dataset/cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl"
    else
        exit 1
    fi
fi

# Create timestamped log and workspace directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/distillation_run_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/workspace_${TIMESTAMP}"

echo "Starting distillation training..."
echo "Dataset: $DATASET_PATH"
echo "Student model: Qwen/Qwen3-30B-A3B-Instruct-2507"
echo "Teacher model: Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Log directory: $LOG_DIR"
echo "Workspace directory: $WORKSPACE_DIR"
echo "Configuration:"
echo "  - Batch size: 1"
echo "  - Group size: 4 (4 rollouts per problem)"
echo "  - KL penalty: 0.5 (allow deviation from teacher)"
echo "  - Max tokens: 4096"
echo ""

# Run distillation training
python train_distillation.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  teacher_model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  batch_size=1 \
  group_size=4 \
  learning_rate=1e-4 \
  max_tokens=4096 \
  kl_penalty_coef=0.5 \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  eval_every=10 \
  save_every=10 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Distillation Training Complete!"
echo "====================================================================="
