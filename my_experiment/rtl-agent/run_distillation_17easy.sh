#!/bin/bash
# On-Policy Distillation Training Script for 17 Easy RTL Problems
# Distills Qwen3-235B (teacher) into Qwen3-30B (student)

set -e  # Exit on error

echo "====================================================================="
echo "RTL Agent - On-Policy Distillation (17 Easy Problems)"
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

# Dataset with 17 easy problems
DATASET_PATH="cvdp_17_easy_problems.jsonl"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Create timestamped log and workspace directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/distill_17easy_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/workspace_17easy_${TIMESTAMP}"

echo "Starting distillation training on 17 easy problems..."
echo "Dataset: $DATASET_PATH"
echo "Student model: Qwen/Qwen3-30B-A3B-Instruct-2507"
echo "Teacher model: Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Log directory: $LOG_DIR"
echo "Workspace directory: $WORKSPACE_DIR"
echo ""
echo "Training Configuration:"
echo "  - Total problems: 17 (all easy difficulty)"
echo "  - Batch size: 2 (process 2 problems at a time)"
echo "  - Group size: 4 (4 rollouts per problem for exploration)"
echo "  - KL penalty: 0.5 (balance correctness vs teacher similarity)"
echo "  - Max tokens: 4096"
echo "  - Learning rate: 1e-4"
echo "  - LoRA rank: 32"
echo ""
echo "Expected runtime: ~2-3 hours for full training"
echo "  (17 problems × 4 rollouts × ~2 min/eval + training time)"
echo ""

# Run distillation training
python train_distillation.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  teacher_model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  batch_size=2 \
  group_size=4 \
  learning_rate=1e-4 \
  max_tokens=4096 \
  kl_penalty_coef=0.5 \
  lora_rank=32 \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  eval_every=5 \
  save_every=5 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Distillation Training Complete!"
echo "====================================================================="
echo "Log directory: $LOG_DIR"
echo ""
echo "Quick analysis commands:"
echo "  1. View metrics:"
echo "     cat $LOG_DIR/metrics.jsonl | jq ."
echo ""
echo "  2. Check pass rates:"
echo "     grep 'pass_rate\|tests_passed\|syntax_valid\|format_valid' $LOG_DIR/metrics.jsonl"
echo ""
echo "  3. Monitor training progress:"
echo "     tail -f $LOG_DIR/logs.log"
echo ""
