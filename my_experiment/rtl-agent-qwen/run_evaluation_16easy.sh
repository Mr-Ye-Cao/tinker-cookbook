#!/bin/bash
# Agentic Evaluation on 16 Easy Problems (Remote API)

set -e

echo "====================================================================="
echo "RTL Agent - Agentic Evaluation (16 Easy Problems)"
echo "Mode: Multi-turn agentic (model executes bash commands)"
echo "Model: Remote API Call"
echo "====================================================================="

# Setup environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../rtl-agent/venv" ]; then
    source "../rtl-agent/venv/bin/activate"
elif [ -d "../rtl-agent-iterative/venv" ]; then
    source "../rtl-agent-iterative/venv/bin/activate"
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Set API keys if needed (even for local API, tinker might check existence)
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env 2>/dev/null | grep TINKER_API_KEY | cut -d'=' -f2 || echo 'dummy')}"

DATASET_PATH="cvdp_16_easy_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/eval_16easy_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/eval_workspace_16easy_${TIMESTAMP}"

# Load configuration from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default API Base (can be overridden by env or .env)
API_BASE="${API_BASE:-http://localhost:8000/v1}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-8B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-8B}"
# API_KEY should be in .env or environment
API_KEY="${API_KEY:-EMPTY}"

echo ""
echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Tokenizer: $TOKENIZER_NAME"
echo "  API Base: $API_BASE"
echo "  Dataset: $DATASET_PATH"
echo "  Log Dir: $LOG_DIR"
echo ""

# Check Docker image availability
echo "Checking Docker image availability..."
if ! docker images | grep -q "gpt-oss-20b-agent-base"; then
    echo "Warning: Docker image 'gpt-oss-20b-agent-base:latest' not found locally."
    echo "Evaluation usually requires this image for 'iverilog' and tools."
    echo ""
fi

echo "Starting evaluation..."

python evaluate_distillation_agentic.py \
  student_model="$STUDENT_MODEL" \
  tokenizer_name="$TOKENIZER_NAME" \
  api_base="$API_BASE" \
  api_key="$API_KEY" \
  cvdp_jsonl_path="$DATASET_PATH" \
  workspace_dir="$WORKSPACE_DIR" \
  log_dir="$LOG_DIR" \
  concurrency=2 \
  max_turns=60

echo ""
echo "====================================================================="
echo "Evaluation Complete!"
echo "Logs: $LOG_DIR"
echo "====================================================================="
