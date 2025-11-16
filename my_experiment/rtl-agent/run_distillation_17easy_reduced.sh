#!/bin/bash
# On-Policy Distillation with Reduced Group Size to Avoid API Throttling

set -e

echo "====================================================================="
echo "RTL Agent - On-Policy Distillation (17 Easy, Reduced Group Size)"
echo "====================================================================="
echo ""

source venv/bin/activate
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"

DATASET_PATH="cvdp_17_easy_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/distill_17easy_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/workspace_17easy_${TIMESTAMP}"

echo "Starting distillation training..."
echo "Dataset: $DATASET_PATH"
echo "Student: Qwen/Qwen3-30B-A3B-Instruct-2507"
echo "Teacher: Qwen/Qwen3-235B-A22B-Instruct-2507"
echo "Configuration: batch_size=2, group_size=2 (4 rollouts/batch - reduced to avoid throttling)"
echo ""

# Run with reduced group_size to avoid API throttling
python train_distillation.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  teacher_model=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  batch_size=2 \
  group_size=2 \
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
echo "Training Complete!"
echo "====================================================================="
