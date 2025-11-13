# Multi-Turn Agent Integration with Tinker

This document discusses the difference between single-shot generation (current implementation) and multi-turn agentic approaches (qwen-code style), and how to integrate them with Tinker.

## Table of Contents

- [Background: CVDP's Two Paradigms](#background-cvdps-two-paradigms)
- [Approach 1: Single-Shot Generation (Current)](#approach-1-single-shot-generation-current)
- [Approach 2: Multi-Turn Agent (qwen-code style)](#approach-2-multi-turn-agent-qwen-code-style)
- [Comparison](#comparison)
- [Implementation Details for Approach 2](#implementation-details-for-approach-2)
- [Hybrid Approach](#hybrid-approach)
- [Recommendations](#recommendations)

---

## Background: CVDP's Two Paradigms

CVDP benchmark supports two evaluation paradigms:

### 1. Non-Agentic (Single-Shot)
```python
response = llm_call("Design a Verilog module...")
code = extract_code(response)
result = evaluate_with_harness(code)
```

### 2. Agentic (Multi-Turn with Tool Use)
```python
agent = QwenCodeAgent(task)
agent.run_autonomously()  # Uses tools: read, write, bash, etc.
# Agent iterates: write code → test → debug → rewrite → test...
result = evaluate_final_state()
```

**Our current implementation** uses paradigm #1 (single-shot) because it's simpler and works well with Tinker's standard RL training. This document explores how to integrate paradigm #2 (multi-turn agents).

---

## Approach 1: Single-Shot Generation (Current)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Tinker Training Loop (Current)             │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  1. Tinker Model                               │    │
│  │     ↓                                          │    │
│  │     Generate complete Verilog code (single    │    │
│  │     API call, no intermediate steps)          │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │  2. Extract Code                               │    │
│  │     Parse markdown code block                  │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │  3. Write to File                              │    │
│  │     /code/rtl/design.sv                        │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │  4. Run CVDP Docker Harness                    │    │
│  │     iverilog compile → cocotb tests            │    │
│  └────────────────────────────────────────────────┘    │
│                      ↓                                   │
│  ┌────────────────────────────────────────────────┐    │
│  │  5. Get Reward                                 │    │
│  │     0.1 (format) + 0.3 (syntax) + 1.0 (tests) │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  Episode length: 1 turn                                 │
│  Training time per episode: ~1-2 minutes                │
└─────────────────────────────────────────────────────────┘
```

### Characteristics

**Environment Type**: Single-turn, non-interactive

**Episode Structure**:
- 1 observation (task prompt)
- 1 action (generated code)
- 1 reward (evaluation result)
- Episode done

**What the Model Generates**:
```verilog
// Complete Verilog/SystemVerilog module in one shot
module fixed_priority_arbiter(...);
  // Full implementation
endmodule
```

### Pros ✅

1. **Simple Implementation**
   - Straightforward environment design
   - No tool parsing needed
   - Works with existing Tinker RL/distillation code

2. **Fast Training**
   - 1 turn per episode (vs. 10-20 turns for multi-turn)
   - Quick iteration cycles
   - Lower compute cost

3. **Proven Approach**
   - On-policy distillation works well with single-shot
   - Similar to how supervised fine-tuning works
   - Established baselines

4. **Credit Assignment is Clear**
   - Reward directly corresponds to generated code
   - No ambiguity about which action deserves credit

5. **Already Implemented**
   - Ready to use (cvdp_env.py, train_rl.py, train_distillation.py)
   - Tested and documented

### Cons ❌

1. **No Iterative Debugging**
   - Model can't test its code and fix bugs
   - One chance to get it right
   - Can't learn from intermediate failures

2. **No Tool Use**
   - Can't read files dynamically
   - Can't run commands
   - Can't explore the workspace

3. **Less Realistic**
   - Doesn't match how human engineers work
   - Doesn't match CVDP's agentic agents (qwen-code)

4. **Limited Learning**
   - Can't learn "debugging strategies"
   - Can't learn "when to test"
   - Only learns end-to-end generation

### Best For

- Quick baselines
- Distillation from strong single-shot models
- When training time/compute is limited
- When model already generates good code in one shot

### Implementation

See files:
- `cvdp_env.py` - Current implementation
- `train_rl.py` - RL training script
- `train_distillation.py` - Distillation script

---

## Approach 2: Multi-Turn Agent (qwen-code style)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 Multi-Turn Tinker Training Loop                   │
│                                                                    │
│  Turn 1:                                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Model: "I'll read the specification first"            │     │
│  │ Tool Call: file_read("/code/docs/spec.md")            │     │
│  └────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Tool Result: <spec content>                            │     │
│  │ Reward: 0.0 (intermediate)                             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Turn 2:                                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Model: "Now I'll write the RTL"                       │     │
│  │ Tool Call: file_write("/code/rtl/design.sv", <code>)  │     │
│  └────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Tool Result: "File written successfully"               │     │
│  │ Reward: 0.0 (intermediate)                             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Turn 3:                                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Model: "Let me compile it"                            │     │
│  │ Tool Call: bash("iverilog /code/rtl/design.sv")       │     │
│  └────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Tool Result: "Syntax error on line 42..."             │     │
│  │ Reward: 0.0 (intermediate)                             │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Turn 4:                                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Model: "I need to fix the syntax error"               │     │
│  │ Tool Call: file_edit("/code/rtl/design.sv", ...)      │     │
│  └────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ... continues until agent says "done" or max turns ...          │
│                                                                    │
│  Final Turn:                                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Model: "Task complete"                                 │     │
│  │ No tool call                                           │     │
│  └────────────────────────────────────────────────────────┘     │
│                           ↓                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ Run CVDP Harness → Final Reward                       │     │
│  │ Reward: 0.1 + 0.3 + 1.0 = 1.4                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                    │
│  Episode length: 10-20 turns                                      │
│  Training time per episode: ~10-30 minutes                        │
└──────────────────────────────────────────────────────────────────┘
```

### Characteristics

**Environment Type**: Multi-turn, interactive

**Episode Structure**:
- Initial observation: Task prompt + system prompt with available tools
- Multiple turns: Model proposes tool call → Tool executes → Model sees result
- Final reward: Only at the end (sparse)
- Episode done: When model says "done" or max turns reached

**Available Tools**:
- `file_read(path)` - Read file contents
- `file_write(path, content)` - Write file
- `file_edit(path, old, new)` - Edit specific part of file
- `bash(command)` - Run shell command
- `list_files(dir)` - List directory contents

**Tool Call Format** (Example):
```xml
<tool_call>
<tool>file_read</tool>
<path>/code/docs/spec.md</path>
</tool_call>
```

Or JSON format:
```json
{"tool": "file_read", "path": "/code/docs/spec.md"}
```

### Pros ✅

1. **Iterative Debugging**
   - Model can test code and fix bugs
   - Can learn from intermediate failures
   - Mirrors real development workflow

2. **Tool Use Learning**
   - Learns when to read specs
   - Learns when to test
   - Learns debugging strategies

3. **Matches CVDP Agentic Agents**
   - Exactly replicates qwen-code behavior
   - Can compare with CVDP's agentic baseline
   - More realistic benchmark

4. **Richer Learning Signal**
   - Learns multi-step planning
   - Learns from intermediate tool results
   - Can develop sophisticated strategies

5. **Better Exploration**
   - Can try multiple approaches
   - Can recover from mistakes
   - More sample-efficient (potentially)

### Cons ❌

1. **Much More Complex**
   - Tool parsing and execution
   - Multi-turn trajectory management
   - State management across turns

2. **Slower Training**
   - 10-20x more turns per episode
   - Longer episodes = longer training time
   - Higher compute cost

3. **Credit Assignment Problem**
   - Which turn deserves the reward?
   - Hard to attribute success to specific actions
   - May need advanced RL algorithms (e.g., attention-based value functions)

4. **Requires Tool-Use Training**
   - Model needs to learn tool formats
   - May need pre-training on tool use
   - Higher sample complexity initially

5. **Not Implemented Yet**
   - Requires significant development
   - Needs testing and debugging
   - Unknown performance characteristics

### Best For

- Research on tool-use + RL
- When you need iterative debugging capabilities
- Replicating qwen-code agent behavior exactly
- Long-horizon reasoning tasks
- When you have ample compute resources

### Implementation (Conceptual)

```python
class CVDPMultiTurnEnv(Env):
    """Multi-turn agent environment with tool use"""
    
    def __init__(self, max_turns=20, ...):
        self.max_turns = max_turns
        self.current_turn = 0
        self.conversation_history = []
        self.tools = {
            "file_read": self._tool_file_read,
            "file_write": self._tool_file_write,
            "file_edit": self._tool_file_edit,
            "bash": self._tool_bash,
            "list_files": self._tool_list_files,
        }
    
    async def initial_observation(self):
        system_prompt = """You are an expert hardware designer.
Available tools:
- file_read(path): Read file
- file_write(path, content): Write file
- bash(cmd): Run command
- file_edit(path, old, new): Edit file

Use tools by outputting:
<tool_call>
<tool>tool_name</tool>
<arg_name>value</arg_name>
</tool_call>

When done, output: <done>"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.prompt}
        ]
        return self.renderer.build_generation_prompt(messages)
    
    async def step(self, action: Action):
        # Parse model's response
        message, _ = self.renderer.parse_response(action)
        text = message["content"]
        
        # Check if done
        if "<done>" in text or self.current_turn >= self.max_turns:
            return await self._evaluate_final()
        
        # Parse tool call
        tool_call = self._parse_tool_call(text)
        if tool_call is None:
            # No valid tool call, return error
            return self._tool_error("Invalid tool call format")
        
        # Execute tool
        tool_result = await self._execute_tool(tool_call)
        self.current_turn += 1
        
        # Build next observation
        self.conversation_history.extend([
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Tool result:\n{tool_result}"}
        ])
        
        next_obs = self.renderer.build_generation_prompt(
            [{"role": "system", "content": system_prompt}] +
            self.conversation_history
        )
        
        return StepResult(
            reward=0.0,  # Intermediate reward (or small step reward)
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            metrics={"turn": self.current_turn}
        )
    
    async def _evaluate_final(self):
        """Run CVDP harness and return final reward"""
        eval_result = await self._run_cvdp_evaluation(...)
        final_reward = self._calculate_reward(eval_result)
        
        return StepResult(
            reward=final_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "final_turn": self.current_turn,
                **eval_result
            }
        )
    
    def _parse_tool_call(self, text):
        """Parse <tool_call>...</tool_call> from text"""
        # XML parsing or JSON parsing
        ...
    
    async def _execute_tool(self, tool_call):
        """Execute the tool and return result"""
        tool_name = tool_call["tool"]
        if tool_name in self.tools:
            return await self.tools[tool_name](tool_call)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    
    async def _tool_file_read(self, args):
        path = args["path"]
        try:
            with open(path, 'r') as f:
                content = f.read()
            return f"File content:\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def _tool_file_write(self, args):
        path = args["path"]
        content = args["content"]
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    async def _tool_bash(self, args):
        cmd = args["command"]
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
        except Exception as e:
            return f"Error running command: {e}"
```

---

## Comparison

| Feature | Approach 1 (Single-Shot) | Approach 2 (Multi-Turn) |
|---------|-------------------------|-------------------------|
| **Turns per episode** | 1 | 10-20 |
| **Episode duration** | 1-2 min | 10-30 min |
| **Training speed** | ✅ Fast | ❌ 10x slower |
| **Complexity** | ✅ Simple | ❌ Complex |
| **Tool use** | ❌ No | ✅ Yes |
| **Iterative debugging** | ❌ No | ✅ Yes |
| **Credit assignment** | ✅ Clear | ❌ Difficult |
| **Matches CVDP agentic** | ❌ No | ✅ Yes |
| **Implementation status** | ✅ Done | ❌ Not implemented |
| **RL algorithm** | Standard (IS, PPO) | May need advanced |
| **Best for** | Baselines, distillation | Research, tool-use |

### Training Time Estimates

**Approach 1 (Single-Shot)**:
```
100 problems × 4 rollouts × 1.5 min = 10 hours per epoch
```

**Approach 2 (Multi-Turn)**:
```
100 problems × 4 rollouts × 15 turns × 1.5 min = 150 hours per epoch
```

Note: Multi-turn may actually be faster if episodes terminate early when tests pass.

---

## Implementation Details for Approach 2

### Challenge 1: Tool Call Parsing

**Problem**: Model needs to output tool calls in a parseable format

**Solutions**:

1. **XML Format** (Claude-style):
```xml
<tool_call>
<tool>file_read</tool>
<path>/code/docs/spec.md</path>
</tool_call>
```

2. **JSON Format** (Function calling):
```json
{"tool": "file_read", "args": {"path": "/code/docs/spec.md"}}
```

3. **Natural Language + Parsing**:
```
I'll read the specification file at /code/docs/spec.md
→ Parser extracts: tool=file_read, path=/code/docs/spec.md
```

**Recommendation**: Use XML or JSON for reliability.

### Challenge 2: Credit Assignment

**Problem**: Final reward only given at end, hard to know which action helped

**Solutions**:

1. **Sparse Reward Only** (simplest):
   - All turns get 0.0 reward except last
   - Rely on RL algorithm to propagate credit backwards

2. **Intermediate Rewards**:
   - Small positive reward for successful tool execution (+0.01)
   - Small negative reward for errors (-0.01)
   - Large reward at end for test passing

3. **Hindsight Relabeling**:
   - After episode, relabel intermediate steps with partial credit
   - E.g., if final reward is 1.0, earlier correct actions get 0.1 each

4. **Value Function with Attention**:
   - Train value function to predict future reward
   - Use attention to identify important turns

**Recommendation**: Start with sparse reward + PPO's advantage estimation.

### Challenge 3: State Management

**Problem**: Need to track conversation history, files modified, etc.

**Solution**:
```python
class EnvironmentState:
    conversation_history: List[Message]
    files_modified: Set[str]
    commands_run: List[str]
    current_turn: int
    workspace_snapshot: Dict[str, str]  # For rollback if needed
```

### Challenge 4: Tool Safety

**Problem**: Model might run dangerous commands

**Solutions**:
- Whitelist allowed commands
- Sandbox execution (Docker already provides this)
- Timeout limits
- Resource limits (max file size, etc.)

---

## Hybrid Approach

Combine both approaches for best results:

### Two-Stage Training

**Stage 1: Single-Shot Pre-training** (Weeks 1-2)
```bash
python train_rl.py --cvdp_jsonl_path dataset.jsonl ...
# Or distillation:
python train_distillation.py --student_model gpt-oss-20b ...
```

**Goal**: Teach model to generate reasonable Verilog in one shot

**Expected Results**:
- Format rate: 90%+
- Syntax rate: 60%+
- Pass rate: 20-30%

**Stage 2: Multi-Turn Fine-Tuning** (Weeks 3-4)
```bash
python train_multi_turn.py \
  --load_checkpoint_path stage1_final.pth \
  --enable_tools \
  --max_turns 10
```

**Goal**: Teach model to debug and iterate

**Expected Results**:
- Format rate: 95%+
- Syntax rate: 80%+
- Pass rate: 40-60%

### Benefits of Hybrid Approach

1. **Fast Initial Learning**: Single-shot trains quickly
2. **Builds on Foundation**: Multi-turn fine-tuning starts from competent model
3. **Sample Efficient**: Less multi-turn data needed
4. **Easier Debug**: Can identify if issues are generation vs. debugging
5. **Benchmarking**: Can compare single-shot vs. multi-turn on same base model

---

## Recommendations

### For Most Users: Start with Approach 1 ✅

**If you want to:**
- Get quick baselines
- Try distillation
- Have limited compute
- Replicate paper results

**Then use:**
```bash
python train_rl.py --cvdp_jsonl_path dataset.jsonl ...
# or
python train_distillation.py --student gpt-oss-20b --teacher qwen3-30b ...
```

### For Researchers: Try Approach 2 🔬

**If you want to:**
- Research tool-use + RL
- Replicate qwen-code exactly
- Study iterative debugging
- Have ample compute

**Then:**
1. Use Approach 1 implementation as reference
2. Extend `CVDPEnv` to support multi-turn
3. Implement tool parsing and execution
4. Start with small-scale experiments (10-20 problems)

### For Best Results: Hybrid Approach 🌟

1. **Week 1-2**: Train with Approach 1
2. **Week 2**: Evaluate and analyze failure modes
3. **Week 3-4**: Add multi-turn capabilities if needed
4. **Week 4+**: Compare both approaches

---

## Code Organization

### Current Implementation (Approach 1)

```
rtl-agent/
├── cvdp_env.py              # Single-shot environment
├── cvdp_dataset.py          # Dataset loaders
├── train_rl.py              # RL training
└── train_distillation.py    # Distillation training
```

### Future Implementation (Approach 2)

```
rtl-agent/
├── cvdp_env.py              # Single-shot environment
├── cvdp_multi_turn_env.py   # Multi-turn environment (NEW)
├── tool_executor.py         # Tool execution (NEW)
├── tool_parser.py           # Tool call parsing (NEW)
├── cvdp_dataset.py          # Dataset loaders
├── train_rl.py              # RL training (single-shot)
├── train_multi_turn_rl.py   # Multi-turn RL (NEW)
└── train_distillation.py    # Distillation training
```

---

## Conclusion

**Approach 1 (Single-Shot)** is:
- ✅ Implemented and ready to use
- ✅ Simple and fast
- ✅ Proven with distillation
- ✅ Good starting point

**Approach 2 (Multi-Turn)** is:
- 🔬 Research-oriented
- 🔬 Matches qwen-code behavior
- 🔬 Requires implementation
- 🔬 Higher complexity/cost

**Recommendation**: **Start with Approach 1**, then add multi-turn capabilities if needed based on results.

---

## References

- **qwen-code**: https://github.com/QwenLM/Qwen-Code
- **CVDP Benchmark**: https://github.com/nvidia/cvdp_benchmark  
- **CVDP Agentic Agents**: See `/qwen3-coder-30b-agent/` and `/gpt-oss-20b-qwen-code-agent/`
- **Tinker RL**: `tinker_cookbook/rl/train.py`
- **On-Policy Distillation**: `tinker_cookbook/distillation/train_on_policy.py`
- **Tool-Use RL Papers**: [ReAct](https://arxiv.org/abs/2210.03629), [Toolformer](https://arxiv.org/abs/2302.04761)

---

**Last Updated**: 2025-01-13  
**Status**: Approach 1 implemented, Approach 2 conceptual
