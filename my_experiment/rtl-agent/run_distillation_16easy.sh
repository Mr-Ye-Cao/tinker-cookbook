#!/bin/bash
# On-Policy Distillation with 16 Easy Problems (divisible by batch_size=2)

set -e

echo "====================================================================="
echo "RTL Agent - On-Policy Distillation (16 Easy Problems)"
echo "====================================================================="

source venv/bin/activate
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"

DATASET_PATH="cvdp_16_easy_problems.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/distill_16easy_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/workspace_16easy_${TIMESTAMP}"

echo "Dataset: $DATASET_PATH (16 problems, divisible by batch_size=2)"
echo "Configuration: batch_size=2, group_size=2"
echo "Expected batches: 16/2 = 8 (all complete, no partial batches)"
echo ""

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
echo "Log directory: $LOG_DIR"
echo "====================================================================="
