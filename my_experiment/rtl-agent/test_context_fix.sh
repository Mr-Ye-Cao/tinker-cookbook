#!/bin/bash
# Quick test script to verify context fix with single rollout

set -e

echo "====================================================================="
echo "Testing Context Fix - Single Rollout"
echo "====================================================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"

DATASET_PATH="cvdp_5_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/test_context_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/test_workspace_${TIMESTAMP}"

echo "Starting test with context fix..."
echo "Dataset: $DATASET_PATH"
echo "Student model: Qwen/Qwen3-30B-A3B-Instruct-2507"
echo "Teacher model: Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Log directory: $LOG_DIR"
echo "Workspace directory: $WORKSPACE_DIR"
echo "Configuration: Single rollout (group_size=1) for quick test"
echo ""

# Run with single rollout for quick testing
python train_distillation.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  teacher_model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  batch_size=1 \
  group_size=1 \
  learning_rate=1e-4 \
  max_tokens=4096 \
  kl_penalty_coef=0.5 \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  eval_every=1 \
  save_every=1 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Test Complete!"
echo "====================================================================="
echo "Log directory: $LOG_DIR"
echo ""
echo "Quick checks:"
echo "1. Check if specification was included in prompt:"
echo "   grep -A 5 'Context: Specification' $LOG_DIR/logs.log"
echo ""
echo "2. Check test results:"
echo "   grep 'tests_passed' $LOG_DIR/metrics.jsonl"
echo ""
echo "3. View full metrics:"
echo "   cat $LOG_DIR/metrics.jsonl | jq ."
echo ""
