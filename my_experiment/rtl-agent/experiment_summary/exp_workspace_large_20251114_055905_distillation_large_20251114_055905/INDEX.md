# RTL Agent Experiment Documentation Index

This directory contains comprehensive documentation of the on-policy distillation experiments conducted on November 14, 2024.

---

## Quick Navigation

### 📊 For Quick Overview
**Start here:** [metrics_summary.md](metrics_summary.md)
- Experiment configurations at a glance
- Reward metrics comparison table
- Performance metrics (timing, tokens)
- Key findings summary

### 📖 For Detailed Analysis
**Deep dive:** [README.md](README.md)
- Full experimental setup and methodology
- Complete analysis of both experiments
- Detailed failure analysis
- Training challenges and recommendations
- Future experiment suggestions

### 🔍 For Code Comparison
**See differences:** [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md)
- Side-by-side code comparison
- Interface mismatches highlighted
- Test case walkthroughs
- Visual priority order comparison
- Root cause analysis

---

## Document Summary

### 1. README.md (Main Report)
**Length:** ~8,000 words
**Sections:**
- Executive Summary
- Experimental Setup
- Experiment 1: Base Model Failure
- Experiment 2: Instruct Model Partial Success
- Student Model Output Analysis
- Test Failure Analysis
- Teacher Model (Reference Solution)
- Training Metrics Analysis
- Key Insights & Patterns
- Training Challenges Identified
- Recommendations for Future Experiments
- Conclusion
- Appendix: File Locations

**Best for:**
- Understanding what happened and why
- Learning from failures
- Planning next experiments
- Complete project documentation

### 2. student_vs_teacher_comparison.md
**Length:** ~3,500 words
**Sections:**
- Interface Comparison
- Priority Logic Comparison
- Test Case Comparison (5 test cases)
- Error Summary (categorized)
- Visual Priority Order Comparison
- Code Quality Assessment
- Root Cause Analysis
- Appendix: Full Side-by-Side Code

**Best for:**
- Understanding specific code differences
- Debugging why tests failed
- Learning what went wrong
- Training model improvement

### 3. metrics_summary.md
**Length:** ~1,500 words
**Sections:**
- Experiment Overview Table
- Reward Metrics
- Performance Metrics
- Detailed Metrics (JSON)
- Configuration Comparison
- Key Findings Summary
- Recommendations
- Success Criteria
- Data Sources

**Best for:**
- Quick reference
- Comparing experiments
- Tracking metrics
- Setting goals for next run

### 4. INDEX.md (This File)
**Purpose:** Navigation and overview

---

## Experiment Results at a Glance

| Aspect | Experiment 1 (Base) | Experiment 2 (Instruct) |
|--------|---------------------|-------------------------|
| **Student** | Qwen3-8B | Qwen3-30B-Instruct |
| **Teacher** | Qwen3-30B | Qwen3-235B-Instruct |
| **Total Reward** | 0.0 / 1.4 ❌ | 0.4 / 1.4 ⚠️ |
| **Format Valid** | ❌ Failed | ✅ Passed |
| **Syntax Valid** | ❌ Failed | ✅ Passed |
| **Tests Passed** | ❌ Failed | ❌ Failed |
| **Time** | 237s | 75s |
| **Tokens** | 4096 (maxed) | 1446 |
| **Status** | Complete failure | Partial success |

---

## Key Findings (TL;DR)

### What Worked ✅
1. Infrastructure works end-to-end
2. Instruct models can format code correctly
3. Generated code compiles successfully
4. CVDP evaluation pipeline functional
5. Both student (30B) and teacher (235B) models loaded successfully

### What Failed ❌
1. Tests failed due to interface mismatches (wrong signal names)
2. Priority order implemented backwards (MSB-first vs LSB-first)
3. Missing required outputs (grant_index)
4. Only 1 training step - no real learning
5. Only 1 problem - insufficient training data

### Main Insight 💡
**The student model knows how to write RTL code, but doesn't know how to follow specifications exactly.**

It generates "textbook example" code using best practices, but fails to match the exact requirements (signal names, priority order, data types).

---

## Problem Description

**Task:** Design a `fixed_priority_arbiter` module in SystemVerilog

**Requirements:**
- 8 request inputs with fixed priority (bit 0 = highest)
- Priority override mechanism (8-bit vector)
- One-hot grant output with valid flag
- Binary grant_index output
- Active-high synchronous reset

**Student Errors:**
- Used wrong signal names (rst_n vs reset, grant_valid vs valid)
- Implemented MSB-first priority (backwards)
- Treated priority_override as boolean instead of vector
- Missing grant_index output entirely
- Used active-low async reset instead of active-high sync

**Result:** Tests cannot even run due to interface mismatches

---

## Recommended Reading Order

### For Time-Pressed Readers (5 minutes)
1. This INDEX.md file
2. [metrics_summary.md](metrics_summary.md) - Key Findings Summary section

### For Technical Understanding (20 minutes)
1. [README.md](README.md) - Executive Summary
2. [README.md](README.md) - Experiment 2 Results
3. [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Interface Comparison
4. [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Test Case 2

### For Complete Analysis (60 minutes)
1. [README.md](README.md) - Full document
2. [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Full document
3. [metrics_summary.md](metrics_summary.md) - Full document

### For Hands-On Investigation
1. Read [README.md](README.md) - Appendix: File Locations
2. Navigate to log directories
3. Inspect config.json, metrics.jsonl, logs.log
4. Explore workspace directories for generated code
5. Cross-reference with documentation

---

## Related Files & Directories

### Experiment Logs
```
../logs/distillation_run_20251114_053608/     # Experiment 1 (base model)
├── config.json
├── metrics.jsonl
├── checkpoints.jsonl
├── logs.log
└── code.diff

../logs/distillation_large_20251114_055905/   # Experiment 2 (instruct model)
├── config.json
├── metrics.jsonl
├── checkpoints.jsonl
├── logs.log
└── code.diff
```

### Workspaces
```
../workspaces/workspace_large_20251114_055905/
├── episode_*/        # Generated code per episode
└── checkpoint_*/     # Model checkpoints
```

### Configuration & Code
```
../run_distillation.sh           # Training launch script
../train_distillation.py         # Training configuration
../cvdp_dataset.py               # CVDP dataset builder
../cvdp_5_problems.jsonl         # Dataset (1 problem loaded)
```

### Dataset Source
```
/home/ubuntu/peter/benchmark/cvdp_benchmark/example_dataset/
└── cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl
```

---

## Questions Answered by These Docs

### General Questions
- **What experiments were run?** → [README.md](README.md) - Experimental Setup
- **What were the results?** → [metrics_summary.md](metrics_summary.md) - Reward Metrics
- **Why did tests fail?** → [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Test Failure Analysis

### Technical Questions
- **What code did the student generate?** → [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Student Model Output
- **What should it have generated?** → [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Teacher Model
- **What were the specific errors?** → [student_vs_teacher_comparison.md](student_vs_teacher_comparison.md) - Error Summary

### Metrics Questions
- **How long did training take?** → [metrics_summary.md](metrics_summary.md) - Performance Metrics
- **How many tokens were used?** → [metrics_summary.md](metrics_summary.md) - Token Usage
- **What was the KL divergence?** → [metrics_summary.md](metrics_summary.md) - Model Behavior

### Planning Questions
- **What should we try next?** → [README.md](README.md) - Recommendations
- **How do we improve?** → [README.md](README.md) - Training Challenges
- **What are success criteria?** → [metrics_summary.md](metrics_summary.md) - Success Criteria

---

## How to Use This Documentation

### For Understanding Past Experiments
1. Read the summaries to understand what happened
2. Review metrics to quantify performance
3. Study code comparisons to see specific errors
4. Use this knowledge to avoid repeating mistakes

### For Planning Future Experiments
1. Review recommendations in README.md
2. Check success criteria in metrics_summary.md
3. Use identified challenges as a checklist
4. Reference configurations that worked/failed

### For Debugging
1. Compare student output to teacher (student_vs_teacher_comparison.md)
2. Check exact error messages in original logs
3. Trace through test cases to understand failures
4. Verify configurations match expectations

### For Reporting
1. Use metrics tables for presentations
2. Reference specific sections for detailed explanations
3. Include code comparisons to show challenges
4. Cite recommendations for next steps

---

## Document Metadata

**Created:** November 14, 2024
**Authors:** Claude Code (Analysis), Human (Experimental Setup)
**Experiments Covered:** 2
- distillation_run_20251114_053608 (Base model)
- distillation_large_20251114_055905 (Instruct model)

**Problem:** Fixed Priority Arbiter (cvdp_agentic_fixed_arbiter_0001)
**Dataset:** CVDP v1.0.1 (1 problem loaded)
**Evaluation:** Docker-based Verilator/Icarus + Cocotb tests

**Document Status:** Complete ✅
**Last Updated:** November 14, 2024
**Version:** 1.0

---

## Quick Links

- [Main Analysis Report](README.md)
- [Code Comparison](student_vs_teacher_comparison.md)
- [Metrics Summary](metrics_summary.md)
- [Experiment 1 Logs](../logs/distillation_run_20251114_053608/)
- [Experiment 2 Logs](../logs/distillation_large_20251114_055905/)
- [Experiment 2 Workspace](../workspaces/workspace_large_20251114_055905/)

---

**Happy Reading! 📚**

For questions or clarifications, refer to the detailed sections in each document.
