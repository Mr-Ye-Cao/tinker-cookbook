#!/bin/bash
# Agentic On-Policy Distillation with 1 Problem (Debugging)

set -e

echo "====================================================================="
echo "RTL Agent - Agentic On-Policy Distillation (1 Problem)"
echo "Teacher: openai/gpt-oss-120b → Student: openai/gpt-oss-20b"
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
LOG_DIR="./logs/agentic_distill_1problem_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/agentic_workspace_1problem_${TIMESTAMP}"

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_PATH (1 problem)"
echo "  Mode: Multi-turn agentic (max 50 turns per episode)"
echo "  Batch size: 1 (single problem)"
echo "  Group size: 2 (GRPO-style)"
echo "  Max tokens: 4096 (per turn)"
echo "  KL penalty coefficient: 0.7"
echo "  Docker image: gpt-oss-20b-agent-base:latest"
echo ""

# Check Docker image availability
echo "Checking Docker image availability..."
if ! docker images | grep -q "gpt-oss-20b-agent-base"; then
    echo "Error: Docker image 'gpt-oss-20b-agent-base:latest' not found!"
    echo "Please build it first:"
    echo "  cd /home/ubuntu/peter/benchmark/cvdp_benchmark/gpt-oss-20b-qwen-code-agent"
    echo "  ./build.sh"
    exit 1
fi
echo "Docker image found locally."

echo ""
echo "Starting training..."
echo ""

python train_distillation_agentic.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=openai/gpt-oss-20b \
  teacher_model=openai/gpt-oss-120b \
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
