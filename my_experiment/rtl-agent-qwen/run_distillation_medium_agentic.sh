#!/bin/bash
# Agentic On-Policy Distillation with Medium Difficulty Problems (Qwen Version)

set -e

echo "====================================================================="
echo "RTL Agent - Agentic On-Policy Distillation (Medium Problems - Qwen)"
echo "Teacher: Qwen/Qwen3-32B -> Student: Qwen/Qwen3-8B"
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

# Set API keys
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env 2>/dev/null | grep TINKER_API_KEY | cut -d'=' -f2 || echo '')}"

# Get WANDB_API_KEY from netrc if not already set
if [ -z "$WANDB_API_KEY" ]; then
    WANDB_API_KEY="$(grep -A2 'api.wandb.ai' ~/.netrc 2>/dev/null | grep password | awk '{print $2}')"
    export WANDB_API_KEY
fi

if [ -z "$TINKER_API_KEY" ]; then
    echo "Error: TINKER_API_KEY not set"
    echo "Please set TINKER_API_KEY environment variable or create ../../.env file"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not set - wandb logging will be disabled"
    echo "Run 'wandb login' to enable wandb logging"
else
    echo "WANDB_API_KEY found (length: ${#WANDB_API_KEY}) - wandb logging enabled"
fi

DATASET_PATH="cvdp_medium_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/qwen_agentic_distill_medium_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/qwen_agentic_workspace_medium_${TIMESTAMP}"

echo ""
echo "Configuration:"
echo "  Student: Qwen/Qwen3-8B"
echo "  Teacher: Qwen/Qwen3-32B"
echo "  Dataset: $DATASET_PATH (55 medium problems)"
echo "  Mode: Multi-turn agentic (max 60 turns per episode)"
echo "  Batch size: 2"
echo "  Group size: 2 (GRPO-style)"
echo "  Max tokens: 12000 (per turn)"
echo "  Command timeout: 30 seconds (per command)"
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
  student_model=Qwen/Qwen3-8B \
  teacher_model=Qwen/Qwen3-32B \
  batch_size=2 \
  group_size=2 \
  learning_rate=1e-4 \
  max_tokens=12000 \
  renderer_name=qwen3_keep_thinking \
  max_turns=60 \
  kl_penalty_coef=0.7 \
  lora_rank=32 \
  docker_image=gpt-oss-20b-agent-base:latest \
  timeout_seconds=30 \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  wandb_project=rtl-agent-qwen \
  eval_every=5 \
  save_every=5 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Training Complete!"
echo "Log directory: $LOG_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "====================================================================="
