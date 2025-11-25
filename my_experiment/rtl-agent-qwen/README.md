# RTL Agent - Qwen Version

Agentic on-policy distillation for RTL code generation using Qwen models.

## Overview

This is a Qwen-adapted version of the RTL agent distillation setup. The key differences from the GPT-OSS version:

1. **Models**: Uses Qwen3 models instead of GPT-OSS
   - Student: `Qwen/Qwen3-8B` (smallest available)
   - Teacher: `Qwen/Qwen3-32B`

   Available models on tinker: `Qwen3-8B`, `Qwen3-30B-A3B` (MoE), `Qwen3-32B`, `Qwen3-235B-Instruct-2507`

2. **Renderer**: Uses standard tinker renderers (`qwen3`, `qwen3_disable_thinking`) instead of Harmony encoding

3. **Tool Calling Format**: Uses text-based `<tool_call>` tags instead of Harmony token-level parsing:
   ```
   <tool_call>
   {"name": "execute_bash", "args": {"command": "ls -la"}}
   </tool_call>
   ```

## Files

- `cvdp_agentic_env_qwen.py` - Environment that uses standard tinker renderers and text-based command extraction
- `cvdp_agentic_dataset.py` - Dataset loader for CVDP problems
- `train_distillation_agentic.py` - Training script for Qwen models
- `utils.py` - Helper functions
- `AGENTIC_SYSTEM_MESSAGE.txt` - System message teaching tool usage (Qwen format)
- `run_distillation_1problem_agentic.sh` - Shell script to run training on 1 problem
- `cvdp_1_problem.jsonl` - Single problem dataset for debugging

## Usage

1. Ensure you have the tinker API key set:
   ```bash
   export TINKER_API_KEY=your_key_here
   ```

2. Run the training:
   ```bash
   ./run_distillation_1problem_agentic.sh
   ```

3. Or run directly with Python:
   ```bash
   python train_distillation_agentic.py \
     student_model=Qwen/Qwen3-8B \
     teacher_model=Qwen/Qwen3-32B \
     cvdp_jsonl_path=cvdp_1_problem.jsonl
   ```

## How It Works

1. **Environment Setup**: Creates a Docker container with HDL tools (iverilog, pytest, cocotb)

2. **Conversation**: Model receives the RTL task and uses `<tool_call>` format to execute bash commands

3. **Command Extraction**: The environment parses `<tool_call>` tags from model output:
   - Primary: `<tool_call>{"name": "execute_bash", "args": {"command": "..."}}</tool_call>`
   - Fallback: ` ```bash ... ``` ` code blocks

4. **Execution**: Commands run in Docker, results returned as observations

5. **Distillation**: Teacher model provides KL penalty on student's action tokens

## Key Differences from GPT-OSS Version

| Aspect | GPT-OSS Version | Qwen Version |
|--------|-----------------|--------------|
| Encoding | Harmony (openai_harmony) | Standard tinker renderers |
| Tool Format | Native Harmony tokens | Text-based `<tool_call>` tags |
| Models | gpt-oss-20b/120b | Qwen3-4B/32B |
| Parser | StreamableParser | Regex-based text parsing |

## Configuration Options

See `CLIConfig` in `train_distillation_agentic.py` for all options:

- `student_model`: Student model name (default: `Qwen/Qwen3-8B`)
- `teacher_model`: Teacher model name (default: `Qwen/Qwen3-32B`)
- `renderer_name`: Override auto-detected renderer
- `max_turns`: Maximum turns per episode (default: 50)
- `kl_penalty_coef`: KL penalty coefficient (default: 0.7)
- `batch_size`: Problems per batch (default: 1)
- `group_size`: Rollouts per problem (default: 1)
