# Training Metrics Summary

Quick reference for the two experiments conducted on November 14, 2024.

---

## Experiment Overview

| Metric | Experiment 1 (Base) | Experiment 2 (Instruct) |
|--------|---------------------|-------------------------|
| **Student Model** | Qwen/Qwen3-8B | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| **Teacher Model** | Qwen/Qwen3-30B-A3B | Qwen/Qwen3-235B-A22B-Instruct-2507 |
| **Model Renderer** | qwen3 (base) | qwen3_instruct |
| **Dataset** | cvdp_5_problems.jsonl | cvdp_5_problems.jsonl |
| **Problems Loaded** | 1 | 1 |
| **Training Steps** | 1 | 1 |
| **Batch Size** | 1 | 1 |
| **Group Size** | 1 | 1 |
| **Learning Rate** | 1e-4 | 1e-4 |
| **KL Penalty Coef** | 1.0 | 1.0 |
| **Max Tokens** | 4096 | 4096 |

---

## Reward Metrics

### Experiment 1 (Base Model - FAILED)

```
┌─────────────────────┬────────┬─────────┐
│ Metric              │ Value  │ Weight  │
├─────────────────────┼────────┼─────────┤
│ Format Valid        │ 0.0    │ × 0.1   │
│ Syntax Valid        │ 0.0    │ × 0.3   │
│ Tests Passed        │ 0.0    │ × 1.0   │
├─────────────────────┼────────┼─────────┤
│ TOTAL REWARD        │ 0.0    │ / 1.4   │
└─────────────────────┴────────┴─────────┘

Pass Rate: 0.0%
```

**Failure Reason:** No valid Verilog code block found in response

### Experiment 2 (Instruct Model - PARTIAL SUCCESS)

```
┌─────────────────────┬────────┬─────────┐
│ Metric              │ Value  │ Weight  │
├─────────────────────┼────────┼─────────┤
│ Format Valid        │ 1.0 ✅ │ × 0.1   │
│ Syntax Valid        │ 1.0 ✅ │ × 0.3   │
│ Tests Passed        │ 0.0 ❌ │ × 1.0   │
├─────────────────────┼────────┼─────────┤
│ TOTAL REWARD        │ 0.4    │ / 1.4   │
└─────────────────────┴────────┴─────────┘

Pass Rate: 0.0%
```

**Progress:** Format and syntax validation passed, functional tests failed

---

## Performance Metrics

### Token Usage

| Metric | Experiment 1 | Experiment 2 | Improvement |
|--------|--------------|--------------|-------------|
| **Action Tokens/Turn** | 4096 (maxed) | 1446 | -65% 🔽 |
| **Observation Tokens/Turn** | 128 | 126 | Similar |
| **Total Episodes** | 1 | 1 | Same |
| **Turns per Episode** | 1 | 1 | Same |

**Analysis:** Instruct model is 3x more efficient (1446 vs 4096 tokens)

### Timing

| Metric | Experiment 1 | Experiment 2 | Improvement |
|--------|--------------|--------------|-------------|
| **Sampling Time** | 213.2s | 44.1s | -79% 🔽 |
| **Training Time** | 23.8s | 30.9s | +30% 🔼 |
| **Total Time** | 237.0s | 75.0s | -68% 🔽 |

**Analysis:** Instruct model is 3.2x faster overall

### Model Behavior

| Metric | Experiment 1 | Experiment 2 | Interpretation |
|--------|--------------|--------------|----------------|
| **Teacher KL** | 0.2927 | 0.3958 | Higher divergence from teacher |
| **Entropy** | 0.541 | 0.324 | More confident predictions |

**Analysis:**
- Lower entropy in Exp 2 means model is more confident in its outputs
- Higher KL divergence means student deviates more from teacher's distribution
- Both metrics make sense: instruct model is confident but generates different code than teacher

---

## Detailed Metrics (JSONL)

### Experiment 1 Metrics

```json
{
  "env/all/reward/total": 0.0,
  "env/all/reward/format_valid": 0.0,
  "env/all/reward/syntax_valid": 0.0,
  "env/all/reward/tests_passed": 0.0,
  "env/all/format_valid": 0.0,
  "env/all/syntax_valid": 0.0,
  "env/all/tests_passed": 0.0,
  "env/all/pass_rate": 0.0,

  "teacher_kl": 0.29273661971092224,
  "optim/entropy": 0.5408518910408020,
  "optim/kl_sample_train_v1": 0.0,
  "optim/kl_sample_train_v2": 0.0,

  "env/all/ac_tokens_per_turn": 4096.0,
  "env/all/ob_tokens_per_turn": 128.0,
  "env/all/total_episodes": 1.0,
  "env/all/turns_per_episode": 1.0,

  "time/sample": 213.1945617198944,
  "time/train": 23.760355472564697,
  "time/total": 236.95491719245911,

  "step": 0
}
```

### Experiment 2 Metrics

```json
{
  "env/all/reward/total": 0.4000000059604645,
  "env/all/reward/format_valid": 0.10000000149011612,
  "env/all/reward/syntax_valid": 0.30000001192092896,
  "env/all/reward/tests_passed": 0.0,
  "env/all/format_valid": 1.0,
  "env/all/syntax_valid": 1.0,
  "env/all/tests_passed": 0.0,
  "env/all/pass_rate": 0.0,

  "teacher_kl": 0.39579829573631287,
  "optim/entropy": 0.32386109232902527,
  "optim/kl_sample_train_v1": 0.06311421841382980,
  "optim/kl_sample_train_v2": 0.22024990618228912,

  "env/all/ac_tokens_per_turn": 1446.0,
  "env/all/ob_tokens_per_turn": 126.0,
  "env/all/total_episodes": 1.0,
  "env/all/turns_per_episode": 1.0,

  "time/sample": 44.09682416915894,
  "time/train": 30.884503841400146,
  "time/total": 75.1,

  "step": 0
}
```

---

## Configuration Comparison

### Model Configurations

#### Experiment 1
```json
{
  "student_model": "Qwen/Qwen3-8B",
  "teacher_model": "Qwen/Qwen3-30B-A3B",
  "renderer_name": "qwen3",
  "lora_rank": 32,
  "learning_rate": 0.0001,
  "max_tokens": 4096,
  "kl_penalty_coef": 1.0,
  "kl_discount_factor": 0.0
}
```

#### Experiment 2
```json
{
  "student_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
  "teacher_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "renderer_name": "qwen3_instruct",
  "lora_rank": 32,
  "learning_rate": 0.0001,
  "max_tokens": 4096,
  "kl_penalty_coef": 1.0,
  "kl_discount_factor": 0.0
}
```

### Dataset Configurations

Both experiments used identical CVDP configurations:

```json
{
  "cvdp_jsonl_path": "cvdp_5_problems.jsonl",
  "workspace_dir": "./workspaces/workspace_*",
  "oss_sim_image": "ghcr.io/hdl/sim/osv:latest",
  "timeout_seconds": 300,
  "format_coef": 0.1,
  "syntax_coef": 0.3,
  "test_coef": 1.0,
  "batch_size": 1,
  "group_size": 1
}
```

### Training Configurations

```json
{
  "loss_fn": "importance_sampling",
  "num_substeps": 1,
  "compute_post_kl": false,
  "eval_every": 5,
  "save_every": 5,
  "behavior_if_log_dir_exists": "delete"
}
```

---

## Key Findings Summary

### 🔴 Critical Issues

1. **Only 1 training step executed** - No real training occurred
2. **Only 1 problem in dataset** - Insufficient for learning
3. **Tests failed due to interface mismatches** - Student code incompatible with test harness
4. **Wrong priority order** - Student implements MSB-first instead of LSB-first
5. **Missing outputs** - grant_index output not implemented

### 🟡 Important Insights

1. **Instruct models mandatory** - Base models cannot format output correctly (0.0 vs 0.4 reward)
2. **Efficiency gains** - Instruct model 3x faster and uses 65% fewer tokens
3. **Specification compliance challenge** - Model knows RTL but doesn't follow specs exactly
4. **KL divergence moderate** - Student deviates from teacher (0.396)
5. **Low entropy = high confidence** - Student is confident but wrong

### 🟢 Successes

1. ✅ **Infrastructure works** - End-to-end pipeline functional
2. ✅ **Format validation passes** - Instruct model produces extractable code
3. ✅ **Syntax validation passes** - Generated code compiles successfully
4. ✅ **CVDP integration works** - Docker-based evaluation runs correctly
5. ✅ **Models load successfully** - Both 30B student and 235B teacher operational

---

## Recommendations Based on Metrics

### Immediate Actions

1. **Increase dataset size**
   - Current: 1 problem
   - Target: 5-100 problems
   - Impact: Enable generalization

2. **Increase training steps**
   - Current: 1 step
   - Target: 50-100 steps
   - Impact: Allow actual learning

3. **Increase exploration**
   - Current: group_size=1
   - Target: group_size=4
   - Impact: Try multiple solutions per problem

4. **Reduce KL penalty**
   - Current: kl_penalty_coef=1.0
   - Target: kl_penalty_coef=0.5
   - Impact: Allow more deviation from teacher to learn task rewards

### Monitoring Targets

During improved training, watch these metrics:

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| **Format Valid** | 1.0 | 1.0 | ✅ Met |
| **Syntax Valid** | 1.0 | 1.0 | ✅ Met |
| **Tests Passed** | >0.5 | 0.0 | ❌ Need improvement |
| **Total Reward** | >1.0 | 0.4 | ❌ Need improvement |
| **Teacher KL** | <0.3 | 0.396 | ⚠️ Could be lower |
| **Tokens/Turn** | <2000 | 1446 | ✅ Efficient |

### Success Criteria for Next Experiment

**Minimum Success:**
- [ ] Tests passed reward > 0.0 (at least some tests passing)
- [ ] Total reward > 0.6 (improvement from 0.4)
- [ ] Training runs for >10 steps
- [ ] At least 5 problems in dataset

**Good Success:**
- [ ] Tests passed reward > 0.5 (half of tests passing)
- [ ] Total reward > 1.0 (most stages passing)
- [ ] Visible improvement trend over steps
- [ ] Teacher KL decreasing (student learning to imitate)

**Excellent Success:**
- [ ] Tests passed reward > 0.8 (most tests passing)
- [ ] Total reward > 1.3 (near-perfect)
- [ ] Consistent improvement over training
- [ ] Student matches teacher quality

---

## Data Sources

### Experiment 1
- **Config:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_run_20251114_053608/config.json`
- **Metrics:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_run_20251114_053608/metrics.jsonl`
- **Logs:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_run_20251114_053608/logs.log`

### Experiment 2
- **Config:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_large_20251114_055905/config.json`
- **Metrics:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_large_20251114_055905/metrics.jsonl`
- **Logs:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/logs/distillation_large_20251114_055905/logs.log`
- **Workspace:** `/home/ubuntu/peter/tinker-cookbook/my_experiment/rtl-agent/workspaces/workspace_large_20251114_055905/`

---

**Document Version:** 1.0
**Last Updated:** November 14, 2024
**Companion Documents:**
- README.md (detailed analysis)
- student_vs_teacher_comparison.md (code comparison)
