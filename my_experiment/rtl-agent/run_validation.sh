#!/bin/bash
cd /Users/yecao/Downloads/project/tinker-cookbook/my_experiment/rtl-agent

# Load Tinker API key from parent directory
export $(cat ../../.env | grep TINKER_API_KEY | xargs)

/opt/homebrew/Caskroom/miniforge/base/envs/tinker-notebook/bin/python train_distillation.py \
  cvdp_jsonl_path=cvdp_5_problems.jsonl \
  student_model=openai/gpt-oss-20b \
  teacher_model=Qwen/Qwen3-32B \
  batch_size=2 \
  group_size=1 \
  learning_rate=1e-4 \
  max_tokens=4096 \
  kl_penalty_coef=1.0 \
  workspace_dir=./workspaces \
  log_path=./logs \
  eval_every=5 \
  save_every=5 \
  behavior_if_log_dir_exists=delete
