# RTL Code Generation with Tinker + CVDP

This project integrates the CVDP benchmark with Tinker to train language models on RTL (Verilog/SystemVerilog) code generation using reinforcement learning and on-policy distillation.

> **Note**: This implementation uses **single-shot generation** (Approach 1). For discussion of multi-turn agents (qwen-code style), see [MULTI_TURN_AGENTS.md](MULTI_TURN_AGENTS.md).

## Overview

### What We're Building

**Goal**: Distill a large model (`Qwen3-Coder-30B`) into a smaller model (`gpt-oss-20b`) for RTL code generation, using:
1. **CVDP benchmark** for evaluation (Docker-based synthesis + simulation)
2. **Tinker** for distributed LoRA fine-tuning
3. **On-policy distillation** to leverage teacher model's knowledge

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tinker Training Loop                      │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Student    │───▶│     CVDP     │───▶│   Rewards    │  │
│  │ (gpt-oss-20b)│    │  Evaluation  │    │ (0.0 - 1.4)  │  │
│  └──────────────┘    │              │    └──────────────┘  │
│         │            │  Docker:     │            │          │
│         │            │  • Synthesis │            │          │
│         │            │  • Simulate  │            │          │
│         │            │  • Test      │            │          │
│         │            └──────────────┘            │          │
│         │                                        │          │
│         └─────────▶┌──────────────┐◀────────────┘          │
│                    │   Teacher    │                         │
│                    │(Qwen3-Coder) │                         │
│                    │              │                         │
│                    │ Compute KL   │                         │
│                    │ Divergence   │                         │
│                    └──────────────┘                         │
│                                                               │
│         Final Reward = CVDP_reward - λ·KL(student||teacher) │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. CVDP Environment Wrapper (`cvdp_env.py`)

Wraps CVDP's Docker-based evaluation as a Tinker `Env`:

**Input**: RTL design specification (from `prompt.json`)
**Output**: Reward based on:
- Format validity (0.1): Code extraction successful
- Syntax validity (0.3): Verilator/iverilog compiles
- Test passing (1.0): Cocotb functional tests pass

**Key Features**:
- Async Docker execution for parallel evaluation
- Workspace management per problem
- Harness output parsing
- Reward shaping

### 2. Dataset Integration (`cvdp_dataset.py`)

Loads CVDP JSONL format and creates Tinker `RLDataset`:

```python
{
  "id": "cvdp_agentic_fixed_arbiter_0001",
  "prompt": "Design a fixed_priority_arbiter module...",
  "context": {
    "docs/specification.md": "...",
    "verif/testbench.sv": "..."
  },
  "harness": {
    "docker-compose.yml": "...",
    "src/test_runner.py": "..."
  }
}
```

### 3. Training Scripts

**`train_rl.py`**: Standard RL training with CVDP rewards
**`train_distillation.py`**: On-policy distillation (student learns from teacher)

## Findings & Design Decisions

### Performance Considerations

**CVDP Evaluation Speed**:
- Docker startup: ~5-10s
- Verilator/Icarus compilation: ~10-30s
- Cocotb test execution: ~10-60s
- **Total**: 30s - 2min per evaluation

**Training Scale Estimates**:
```
100 problems × 4 rollouts × 1 min avg = ~6.7 hours per epoch
500 problems × 4 rollouts × 1 min avg = ~33 hours per epoch
```

**Optimization Strategies**:
1. **Parallel Docker execution**: Run 8-16 containers simultaneously
2. **Small initial dataset**: Start with 50-100 problems
3. **Batch pipelining**: Overlap sampling with training (Tinker async)
4. **Cached evaluations**: Store results for identical code

### Reward Shaping

**Problem**: CVDP gives sparse binary rewards (pass/fail)

**Solution**: Multi-stage reward shaping
```python
reward = (
    0.1 * format_valid      # Code block extracted correctly
  + 0.3 * syntax_valid      # Verilator lint passed
  + 0.5 * compilation_ok    # iverilog compiled (optional)
  + 1.0 * tests_passed      # All cocotb tests passed
)
```

**Alternative**: Dense rewards via partial test passing
```python
reward = 1.0 * (num_tests_passed / total_tests)
```

### On-Policy Distillation Strategy

**Teacher**: `Qwen3-Coder-30B` (large, high-quality generations)
**Student**: `gpt-oss-20b` (smaller, faster inference)

**Training Process**:
1. **Student samples** trajectory (generates Verilog code)
2. **CVDP evaluates** → sparse reward R_env
3. **Teacher evaluates** student's tokens → teacher logprobs
4. **Compute KL penalty**: KL = log p_student - log p_teacher
5. **Combined reward**: R = R_env - λ·KL

**Hyperparameters**:
- `kl_penalty_coef`: 0.5-1.0 (balance correctness vs. teacher similarity)
- `kl_discount_factor`: 0.0 (no future KL penalty, reward is end-of-episode)
- Learning rate: 1e-4 (standard for LoRA)
- Batch size: 8-16 problems
- Group size: 2-4 rollouts per problem

### Challenges & Solutions

#### Challenge 1: Slow Evaluation

**Problem**: CVDP Docker evaluation takes 1-2 min per sample

**Solutions**:
- ✅ Use Tinker's async capabilities to overlap sampling/training
- ✅ Run multiple Docker containers in parallel (8-16 workers)
- ✅ Start with small dataset (50 problems) for rapid iteration
- ⚠️ Consider proxy rewards (syntax checking) for early training

#### Challenge 2: Sparse Rewards

**Problem**: Binary pass/fail makes credit assignment difficult

**Solutions**:
- ✅ Reward shaping (format, syntax, compilation stages)
- ✅ Parse cocotb output for partial test passing
- ✅ On-policy distillation adds dense KL signal

#### Challenge 3: Long Sequences

**Problem**: RTL code + specifications can be 2k-8k tokens

**Solutions**:
- ✅ Use models with long context (Qwen3-Coder supports 32k)
- ✅ Set `max_tokens=8192` for generation
- ✅ Truncate specs if needed (see `max_prompt_tokens`)

#### Challenge 4: Environment Determinism

**Problem**: Same code always gets same reward (no exploration noise)

**Solutions**:
- ✅ Temperature sampling (set in `TinkerTokenCompleter`)
- ✅ Multiple rollouts per problem (`group_size=4`)
- ✅ Curriculum learning (start with easier problems)

## Implementation Plan

### Phase 1: Core Infrastructure (This PR)

- [x] `cvdp_env.py` - CVDP environment wrapper
- [x] `cvdp_dataset.py` - Dataset loaders and builders
- [x] `config.py` - Configuration dataclasses
- [x] `utils.py` - Helper functions (Docker, parsing)
- [x] `train_rl.py` - RL training script
- [x] `train_distillation.py` - Distillation training script
- [x] `requirements.txt` - Dependencies

### Phase 2: Testing & Validation

- [ ] Unit tests for Docker execution
- [ ] Test on 5 CVDP problems (golden solutions)
- [ ] Verify reward parsing works correctly
- [ ] Check async Docker parallelism
- [ ] Validate Tinker integration (forward/backward passes)

### Phase 3: Small-Scale Training

- [ ] Train on 50 CVDP problems
- [ ] Monitor metrics: reward, KL, format rate, syntax rate
- [ ] Debug any issues with Docker/harness
- [ ] Tune hyperparameters (LR, KL coef, batch size)

### Phase 4: Full-Scale Training

- [ ] Scale to 500 problems
- [ ] Run multi-day training (10-20 epochs)
- [ ] Evaluate on held-out test set
- [ ] Compare student vs. teacher performance

### Phase 5: Optimization

- [ ] Implement result caching
- [ ] Add proxy rewards for faster iteration
- [ ] Experiment with curriculum learning
- [ ] Try different teacher models

## File Structure

```
rtl-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration classes
├── cvdp_env.py              # CVDP environment wrapper
├── cvdp_dataset.py          # Dataset and builders
├── utils.py                 # Helper functions
├── train_rl.py              # RL training script
├── train_distillation.py    # Distillation training script
├── experiments/             # Training configs and logs
│   ├── rl_basic.py
│   └── distillation_basic.py
└── workspaces/              # CVDP evaluation workspaces
    └── .gitignore
```

## Usage

### Training with Standard RL

```bash
# Basic RL training on CVDP
python train_rl.py \
  --cvdp_dataset /path/to/cvdp_v1.0.1_agentic.jsonl \
  --model_name meta-llama/Llama-3.2-1B \
  --batch_size 8 \
  --group_size 4 \
  --learning_rate 1e-4 \
  --log_path /tmp/rtl-rl-training
```

### Training with On-Policy Distillation

```bash
# Distill Qwen3-Coder-30B into gpt-oss-20b
python train_distillation.py \
  --cvdp_dataset /path/to/cvdp_v1.0.1_agentic.jsonl \
  --student_model gpt-oss-20b \
  --teacher_model Qwen/Qwen3-Coder-30B \
  --batch_size 8 \
  --group_size 2 \
  --learning_rate 1e-4 \
  --kl_penalty_coef 1.0 \
  --log_path /tmp/rtl-distillation
```

## Expected Results

### Baseline (No Training)

- Format rate: 60-80% (model generates valid code blocks)
- Syntax rate: 20-40% (code compiles)
- Pass rate: 5-15% (tests pass)

### After RL Training (10 epochs)

- Format rate: 90-95%
- Syntax rate: 60-80%
- Pass rate: 30-50%

### After Distillation (10 epochs)

- Format rate: 95-98% (teacher guidance)
- Syntax rate: 70-85%
- Pass rate: 40-60% (inherits teacher's code quality)

## Next Steps

1. **Run validation**: Test on 5 golden CVDP problems
2. **Profile performance**: Measure Docker overhead and parallelism
3. **Start training**: Run on 50 problems for quick feedback
4. **Iterate**: Tune rewards, hyperparameters, dataset size
5. **Scale up**: Full 500-problem training runs

## References

- [CVDP Benchmark](https://github.com/nvidia/cvdp_benchmark)
- [Tinker Docs](https://tinker-docs.thinkingmachines.ai/)
- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation)
- [CVDP Paper](https://arxiv.org/abs/2506.14074)
