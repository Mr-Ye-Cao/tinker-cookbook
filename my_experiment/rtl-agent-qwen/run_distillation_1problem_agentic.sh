#!/bin/bash
# Agentic On-Policy Distillation with 1 Problem (Qwen Version)

set -e

echo "====================================================================="
echo "RTL Agent - Agentic On-Policy Distillation (Qwen)"
echo "Teacher: Qwen/Qwen3-32B -> Student: Qwen/Qwen3-4B"
echo "Mode: Multi-turn agentic (model executes bash commands)"
echo "====================================================================="

# Setup environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../rtl-agent/venv" ]; then
    source ../rtl-agent/venv/bin/activate
elif [ -d "../rtl-agent-iterative/venv" ]; then
    source ../rtl-agent-iterative/venv/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Set API key
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env 2>/dev/null | grep TINKER_API_KEY | cut -d'=' -f2 || echo '')}"

if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set"
    echo "Please set TINKER_API_KEY environment variable or create ../../.env file"
    exit 1
fi

DATASET_PATH="cvdp_1_problem.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/qwen_agentic_distill_1problem_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/qwen_agentic_workspace_1problem_${TIMESTAMP}"

echo ""
echo "Configuration:"
echo "  Student: Qwen/Qwen3-4B"
echo "  Teacher: Qwen/Qwen3-32B"
echo "  Dataset: $DATASET_PATH (1 problem)"
echo "  Mode: Multi-turn agentic (max 50 turns per episode)"
echo "  Batch size: 1 (single problem)"
echo "  Group size: 1 (single rollout for debugging)"
echo "  Max tokens: 4096 (per turn)"
echo "  KL penalty coefficient: 0.7"
echo "  Docker image: gpt-oss-20b-agent-base:latest"
echo ""

# Check Docker image availability
echo "Checking Docker image availability..."
if ! docker images | grep -q "gpt-oss-20b-agent-base"; then
    echo "Warning: Docker image 'gpt-oss-20b-agent-base:latest' not found locally."
    echo "The training will still proceed but may fail if Docker is needed."
    echo ""
fi

echo ""
echo "Starting training..."
echo ""

python train_distillation_agentic.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=Qwen/Qwen3-4B \
  teacher_model=Qwen/Qwen3-32B \
  batch_size=1 \
  group_size=1 \
  learning_rate=1e-4 \
  max_tokens=4096 \
  max_turns=50 \
  kl_penalty_coef=0.7 \
  lora_rank=32 \
  docker_image=gpt-oss-20b-agent-base:latest \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  eval_every=1 \
  save_every=1 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Training Complete!"
echo "Log directory: $LOG_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "====================================================================="
