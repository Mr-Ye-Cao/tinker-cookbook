#!/bin/bash
# Agentic Evaluation on Medium + Hard Problems (Remote API)

set -e

echo "====================================================================="
echo "RTL Agent - Agentic Evaluation (Medium + Hard Problems)"
echo "Mode: Multi-turn agentic (model executes bash commands)"
echo "Model: Remote API Call"
echo "Dataset: 55 medium + 20 hard = 75 problems"
echo "====================================================================="

# Setup environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found."
fi

# Load configuration from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default API Base (can be overridden)
API_BASE="${API_BASE:-http://localhost:8000/v1}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-8B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-8B}"
API_KEY="${API_KEY:-EMPTY}"

DATASET_PATH="cvdp_medium_hard_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/eval_medium_hard_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/eval_workspace_medium_hard_${TIMESTAMP}"

echo "Starting evaluation..."

python evaluate_distillation_agentic.py \
  student_model="$STUDENT_MODEL" \
  tokenizer_name="$TOKENIZER_NAME" \
  api_base="$API_BASE" \
  api_key="$API_KEY" \
  cvdp_jsonl_path="$DATASET_PATH" \
  workspace_dir="$WORKSPACE_DIR" \
  log_dir="$LOG_DIR" \
  concurrency=8 \
  max_turns=50 \
  max_tokens=12000 \
  timeout_seconds=30 \
  docker_image="gpt-oss-20b-agent-base:latest"

echo ""
echo "====================================================================="
echo "Evaluation Complete!"
echo "Logs: $LOG_DIR"
echo "====================================================================="
