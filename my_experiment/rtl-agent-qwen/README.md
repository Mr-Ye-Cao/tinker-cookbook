# RTL Agent - Agentic On-Policy Distillation

Multi-turn agentic training for RTL code generation using on-policy distillation.

## Overview

This project implements **agentic training** where the student model learns to:
- Execute bash commands (ls, cat, echo, iverilog, pytest, etc.)
- Read specifications from files
- Create verification tests
- Write RTL files iteratively
- Compile and check syntax (iverilog)
- Run tests and read output
- Debug failures and iterate until tests pass

This is in contrast to **one-shot training** (in `../rtl-agent/`) where the model generates complete Verilog code in a single response.

## Key Differences from One-Shot

| Aspect | One-Shot (rtl-agent/) | Agentic (rtl-agent-qwen/) |
|--------|----------------------|---------------------------|
| **Interaction** | Single prompt → single response | Multi-turn conversation with command execution |
| **Model Output** | Complete Verilog code in markdown block | Bash commands (ls, cat, echo, iverilog, pytest) |
| **Iterations** | None (one chance) | Model can debug failures, edit code, re-test |
| **Episode Length** | 1 turn | Up to 50 turns |
| **Training Tokens** | ~2K-4K per episode | ~8K-16K per episode |
| **Environment** | CVDPEnv (single-turn) | CVDPAgenticEnv (multi-turn) |
| **System Message** | ONESHOT_SYSTEM_MESSAGE.txt | AGENTIC_SYSTEM_MESSAGE.txt |

## Architecture

### Components

```
rtl-agent-qwen/
├── cvdp_agentic_env.py          # Multi-turn environment with Docker execution
├── cvdp_agentic_dataset.py      # Dataset builder for agentic training
├── utils.py                      # Command parsing and Docker helpers
├── train_distillation_agentic.py # Training script
├── run_distillation_16easy_agentic.sh # Launch script
├── AGENTIC_SYSTEM_MESSAGE.txt   # System prompt teaching tool usage
├── cvdp_16_easy_problems.jsonl  # Dataset (symlink to ../rtl-agent/)
└── requirements.txt             # Dependencies
```

### Training Flow

1. **Student generates action** (bash command)
   - Example: "Let me read the spec: `cat /code/docs/specification.md`"

2. **Environment executes command** in Docker container
   - Command: `cat /code/docs/specification.md`
   - Returns: File contents as observation

3. **Student sees output, generates next action**
   - Example: "Now I'll write the RTL: `cat > /code/rtl/arbiter.sv << 'EOF' ... EOF`"

4. **Environment executes, returns observation**
   - Continues until tests pass or max turns (50)

5. **Teacher assigns log probs** to student's actions
   - Teacher sees full conversation: [system] [user] [assistant] [user] [assistant] ...
   - Computes log probabilities for each token in student's actions

6. **Loss computation**
   - Advantage = (reward - baseline) - kl_penalty * (student_logprob - teacher_logprob)
   - Only action tokens get gradients (mask=1), observations don't (mask=0)

### How It Works

**On-Policy Distillation:**
- Student model generates multi-turn trajectories by executing commands
- Teacher model computes log probabilities for student's action tokens
- KL penalty encourages student to match teacher's distribution
- Reward signal comes from CVDP evaluation (format + syntax + tests)

**Advantage Calculation:**
```python
advantage = (reward - group_mean_reward) - kl_penalty_coef * reverse_KL
reverse_KL = student_logprob - teacher_logprob
```

**Reward Structure:**
```
format_coef = 0.1   # Valid code extraction
syntax_coef = 0.3   # iverilog compilation succeeds
test_coef = 1.0     # Pass rate on pytest
Max reward = 1.4
```

## Setup

### Prerequisites

1. **Docker**: Required for running HDL simulations
   ```bash
   # Check Docker is installed
   docker --version
   ```

2. **Python Environment**: Create virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Docker Image**: Pull the agentic base image
   ```bash
   docker pull ghcr.io/peter-shen-simpleai/gpt-oss-20b-agent-base:latest
   ```

   This image includes:
   - iverilog (Icarus Verilog)
   - cocotb (Python testbench framework)
   - pytest
   - Other HDL tools

4. **Tinker API Key**: Set environment variable
   ```bash
   export TINKER_API_KEY=your_api_key_here
   ```

   Or create `../../.env`:
   ```
   TINKER_API_KEY=your_api_key_here
   ```

### Dataset

The dataset (`cvdp_16_easy_problems.jsonl`) is symlinked from `../rtl-agent/`. It contains 16 easy CVDP problems for training.

## Usage

### Quick Start

```bash
# Activate environment
source venv/bin/activate  # or use ../rtl-agent/venv if exists

# Run agentic training on 16 easy problems
./run_distillation_16easy_agentic.sh
```

### Custom Training

```bash
python train_distillation_agentic.py \
  cvdp_jsonl_path=cvdp_16_easy_problems.jsonl \
  student_model=openai/gpt-oss-20b \
  teacher_model=openai/gpt-oss-120b \
  batch_size=2 \
  group_size=2 \
  max_tokens=16384 \
  max_turns=50 \
  learning_rate=1e-4 \
  kl_penalty_coef=0.7
```

### Parameters

**Model Configuration:**
- `student_model`: Base model to train (default: `openai/gpt-oss-20b`)
- `teacher_model`: Teacher model for distillation (default: `openai/gpt-oss-120b`)
- `lora_rank`: LoRA rank for student (default: 32)

**Agentic Configuration:**
- `max_turns`: Maximum turns per episode (default: 50)
- `max_tokens`: Max tokens per episode (default: 16384)
- `docker_image`: Docker image for execution (default: gpt-oss-20b-agent-base)
- `timeout_seconds`: Command execution timeout (default: 600)

**Training Hyperparameters:**
- `batch_size`: Problems per batch (default: 2)
- `group_size`: Rollouts per problem (default: 2)
- `learning_rate`: Learning rate (default: 1e-4)
- `kl_penalty_coef`: KL penalty weight (default: 0.7)

**Reward Shaping:**
- `format_coef`: Format validity reward (default: 0.1)
- `syntax_coef`: Syntax validity reward (default: 0.3)
- `test_coef`: Test passing reward (default: 1.0)

## Example Episode

**Turn 1 - Model reads spec:**
```
Assistant: Let me first read the specification to understand the requirements.
`cat /code/docs/specification.md`

User: [spec contents...]
```

**Turn 2 - Model writes RTL:**
```
Assistant: I'll implement the arbiter module:
cat > /code/rtl/arbiter.sv << 'EOF'
module arbiter(
  input wire clk,
  input wire rst,
  ...
);
...
endmodule
EOF

User: [file written]
```

**Turn 3 - Model compiles:**
```
Assistant: Let me compile and check for syntax errors:
`iverilog -g2012 -o /tmp/test.vvp /code/rtl/arbiter.sv`

User: [compilation output]
```

**Turn 4 - Model runs tests:**
```
Assistant: Now I'll run the tests:
`cd /code && pytest -s`

User:
===== test session starts =====
collected 5 items
test_arbiter.py::test_reset PASSED
test_arbiter.py::test_priority PASSED
test_arbiter.py::test_round_robin PASSED
test_arbiter.py::test_fairness PASSED
test_arbiter.py::test_simultaneous PASSED
===== 5 passed in 2.34s =====
[Episode ends - all tests passed!]
```

## Implementation Details

### CVDPAgenticEnv

Multi-turn environment that:
- Starts persistent Docker container per episode
- Parses bash commands from model output
- Executes commands in container
- Returns command output as next observation
- Ends when tests pass or max_turns reached
- Runs CVDP evaluation for final reward

### Command Parsing (utils.py)

Supports multiple command formats:
```python
# Markdown code block
```bash
ls /code/rtl/
```

# Inline code
`cat /code/docs/spec.md`

# Direct command
ls -la /code/rtl/
```

### Safety Validation

Blocks dangerous commands:
- `rm -rf /`
- `mkfs`
- `dd of=/dev/`
- Fork bombs
- etc.

All commands are sandboxed within `/code` directory in Docker container.

### Observation Formatting

Command outputs are formatted for readability:
```
$ ls /code/rtl/
arbiter.sv
design.v

$ iverilog -g2012 arbiter.sv
[compilation output]

[Exit code: 0]
```

## Monitoring

### Logs

Training logs include:
- Episode start/end markers
- Turn-by-turn command execution
- Command outputs (truncated if >8000 chars)
- Compilation results
- Test results
- Final evaluation metrics
- Reward breakdown

### Metrics

Per-episode metrics:
- `format_valid`: Code extraction successful
- `syntax_valid`: iverilog compilation succeeded
- `tests_passed`: All pytest tests passed
- `pass_rate`: Fraction of tests passed
- `turns_used`: Number of turns taken
- `end_reason`: Why episode ended (tests_passed, max_turns, final_answer)

## Troubleshooting

### Docker Container Issues

**Problem:** "Failed to start Docker container"

**Solution:**
```bash
# Check Docker is running
docker ps

# Pull image manually
docker pull ghcr.io/peter-shen-simpleai/gpt-oss-20b-agent-base:latest

# Clean up old containers
docker container prune
```

### Command Parsing Issues

**Problem:** Model output not recognized as command

**Solution:** Check system prompt teaches model to use:
- Markdown code blocks: ````bash ... ````
- Inline code: `` `command` ``
- Direct commands

### Max Turns Reached

**Problem:** Episode always hits max_turns without solving

**Solutions:**
- Increase `max_turns` (default: 50)
- Improve system prompt to guide model better
- Check if model is stuck in loops
- Verify model can read error messages correctly

### Memory Issues

**Problem:** OOM during training

**Solutions:**
- Reduce `batch_size` (multi-turn uses more memory)
- Reduce `max_tokens` (limit episode length)
- Reduce `max_turns` (fewer turns per episode)

## Comparison with One-Shot

### Advantages of Agentic

1. **Error recovery**: Model can debug failures instead of one-shot guess
2. **Iterative refinement**: Read, implement, test, fix, repeat
3. **More realistic**: Matches how human developers work
4. **Better for complex tasks**: Can break down problem into steps

### Disadvantages of Agentic

1. **More tokens**: 4-8x more tokens per episode
2. **Slower training**: Longer episodes, more Docker operations
3. **More complex**: Command parsing, Docker management, multi-turn logic
4. **Harder to debug**: Multi-turn trajectories more complex than single response

## Next Steps

1. **Evaluate on test set**: Run trained model on held-out problems
2. **Compare to one-shot**: Measure pass rate improvement
3. **Ablation studies**: Test different system prompts, reward shaping
4. **Scale up**: Train on more problems, larger models
5. **Analysis**: Study model's debugging strategies, common failure modes

## References

- **CVDP Benchmark**: Hardware design verification benchmark
- **Tinker API**: On-policy RL training framework
- **qwen-code**: Agentic framework (used for reference, not directly in training)
- **gpt-oss-20b**: 20B parameter student model
- **Qwen3-Coder-30B**: 30B parameter teacher model

## License

See parent project for license information.
