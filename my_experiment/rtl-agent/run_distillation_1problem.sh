#!/bin/bash
# On-Policy Distillation with 1 Problem (arithmetic_progression_generator)

set -e

echo "====================================================================="
echo "RTL Agent - On-Policy Distillation (1 Problem - Debug)"
echo "Teacher: openai/gpt-oss-120b → Student: openai/gpt-oss-20b"
echo "Problem: cvdp_agentic_arithmetic_progression_generator_0001"
echo "====================================================================="

source venv/bin/activate
export TINKER_API_KEY="${TINKER_API_KEY:-$(cat ../../.env | grep TINKER_API_KEY | cut -d'=' -f2)}"

DATASET_PATH="cvdp_1_problem.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/distill_1problem_${TIMESTAMP}"
WORKSPACE_DIR="./workspaces/workspace_1problem_${TIMESTAMP}"

echo "Dataset: $DATASET_PATH (1 problem for debugging)"
echo "Configuration: batch_size=1, group_size=1, max_tokens=8192, kl_coef=0.7"
echo "Expected batches: 1/1 = 1"
echo ""

python train_distillation.py \
  cvdp_jsonl_path="$DATASET_PATH" \
  student_model=openai/gpt-oss-20b \
  teacher_model=openai/gpt-oss-120b \
  batch_size=1 \
  group_size=1 \
  learning_rate=1e-4 \
  max_tokens=8192 \
  kl_penalty_coef=0.7 \
  lora_rank=32 \
  workspace_dir="$WORKSPACE_DIR" \
  log_path="$LOG_DIR" \
  eval_every=1 \
  save_every=1 \
  behavior_if_log_dir_exists=delete

echo ""
echo "====================================================================="
echo "Training Complete!"
echo "Log directory: $LOG_DIR"
echo "====================================================================="
