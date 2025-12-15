"""
CVDP Agentic Environment for Tinker - Qwen Version

Multi-turn environment where the model can execute bash commands,
read files, write RTL, compile, test, debug, and iterate until tests pass.

This version uses standard tinker renderers and text-based tool parsing,
compatible with Qwen models.
"""

import asyncio
import os
import shutil
import json
import logging
import time
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import Action, Env, Observation, StepResult

from utils import (
    format_command_observation,
    truncate_output
)

logger = logging.getLogger(__name__)


class CVDPAgenticEnvQwen(Env):
    """
    Multi-turn agentic environment for RTL design tasks - Qwen version.

    Uses standard tinker renderers instead of Harmony encoding,
    and extracts commands from text-based tool call format.

    The model can:
    - Execute bash commands (ls, cat, echo, iverilog, pytest, etc.)
    - Read specifications
    - Write RTL files
    - Compile and test iteratively
    - Debug failures
    - Iterate until all tests pass

    Workflow:
    1. initial_observation() returns task prompt and starts Docker container
    2. Model generates response with tool calls in <tool_call> tags
    3. step() extracts command, executes it, returns output as observation
    4. Repeat until tests pass or max turns reached
    5. Final evaluation using CVDP harness
    """

    def __init__(
        self,
        problem_id: str,
        prompt: str,
        context_files: Dict[str, str],
        harness_config: Dict[str, str],
        workspace_dir: str,
        renderer: renderers.Renderer,
        system_message: str | None = None,
        docker_image: str = "gpt-oss-20b-agent-base:latest",
        timeout_seconds: int = 30,  # Per-command timeout (avoid hanging simulations)
        max_turns: int = 50,
        format_coef: float = 0.1,
        syntax_coef: float = 0.3,
        test_coef: float = 1.0,
        log_path: str | None = None,  # Directory for turn logs (if None, uses workspace_dir)
        difficulty: str = "unknown",  # Problem difficulty: easy/medium/hard
    ):
        """
        Args:
            problem_id: Unique problem identifier
            prompt: RTL design task description
            context_files: Dict of file_path -> content (docs, verif, etc.)
            harness_config: Dict of harness file_path -> content
            workspace_dir: Directory to store CVDP workspace
            renderer: Tinker renderer for tokenization (Qwen3Renderer)
            system_message: System message teaching tool usage
            docker_image: Docker image with HDL tools (iverilog, pytest, etc.)
            timeout_seconds: Command execution timeout
            max_turns: Maximum turns before episode ends
            format_coef: Reward coefficient for valid format
            syntax_coef: Reward coefficient for syntax validity
            test_coef: Reward coefficient for passing tests
            log_path: Directory for turn logs (if None, uses workspace_dir)
            difficulty: Problem difficulty level (easy/medium/hard)
        """
        self.problem_id = problem_id
        self.difficulty = difficulty
        self.prompt = prompt
        self.context_files = context_files
        self.harness_config = harness_config
        self.system_message = system_message
        self.renderer = renderer
        self.max_turns = max_turns
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        
        # Generate unique worker ID for this environment instance
        self.worker_id = str(uuid.uuid4())[:8]
        
        # Setup workspace and logging
        self.workspace_dir = workspace_dir
        self.log_path = log_path

        # Reward coefficients
        self.format_coef = format_coef
        self.syntax_coef = syntax_coef
        self.test_coef = test_coef

        # Episode state
        self.current_turn = 0
        self.conversation_history: List[renderers.Message] = []
        self.docker_container_id: Optional[str] = None
        self.episode_ended = False

        # Context management - Qwen3-8B has 32k context
        self.max_context_tokens = 19000  # Leave room for max_tokens=12000 (19k + 12k = 31k, safe buffer)
        self.keep_first_n_messages = 2   # Always keep system + initial user prompt

        # Per-turn logging directory
        self.turn_logs_dir: Optional[str] = None

        # Setup workspace
        self._setup_workspace()

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _estimate_tokens(self, text: str) -> int:
        """
        Accurate token estimate using renderer's tokenizer.
        """
        try:
            return len(self.renderer.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            # Fallback to conservative estimate (2.5 chars per token for code)
            return int(len(text) / 2.5)

    def _get_context_tokens(self) -> int:
        """Estimate total tokens in conversation history"""
        total = 0
        for msg in self.conversation_history:
            total += self._estimate_tokens(msg.get("content", ""))
        return total

    def _truncate_context_if_needed(self):
        """
        Truncate conversation history if it exceeds max_context_tokens.

        Strategy:
        1. Always keep first N messages (system + initial user prompt)
        2. Keep most recent messages
        3. Drop middle messages if needed
        4. Also truncate individual long tool outputs
        """
        current_tokens = self._get_context_tokens()

        if current_tokens <= self.max_context_tokens:
            return

        self.task_logger.info(f"Context too large ({current_tokens} tokens), truncating...")

        # First, try truncating long tool outputs in recent messages
        for i in range(self.keep_first_n_messages, len(self.conversation_history)):
            msg = self.conversation_history[i]
            content = msg.get("content", "")

            # Truncate very long outputs (likely tool results)
            if len(content) > 4000:
                truncated = content[:2000] + "\n\n[... truncated for context management ...]\n\n" + content[-1000:]
                self.conversation_history[i] = renderers.Message(
                    role=msg["role"],
                    content=truncated
                )

        # Recalculate
        current_tokens = self._get_context_tokens()

        if current_tokens <= self.max_context_tokens:
            self.task_logger.info(f"Context reduced to {current_tokens} tokens after truncating outputs")
            return

        # If still too large, drop older messages (keep first N and recent ones)
        while current_tokens > self.max_context_tokens and len(self.conversation_history) > self.keep_first_n_messages + 2:
            # Remove the oldest message after the initial ones
            removed = self.conversation_history.pop(self.keep_first_n_messages)
            current_tokens = self._get_context_tokens()
            self.task_logger.info(f"Dropped old message, context now {current_tokens} tokens")

        self.task_logger.info(f"Final context size: {current_tokens} tokens, {len(self.conversation_history)} messages")

    def _build_prompt_with_context(self) -> str:
        """
        Build enriched prompt including specification and relevant context files.
        Respects max_context_tokens to prevent overflow.

        Returns:
            Complete prompt with embedded context files
        """
        parts = [self.prompt]
        parts.append("")  # Blank line separator
        
        # Track current token usage
        current_tokens = self._estimate_tokens(self.prompt)
        
        # Account for System Message and Template overhead
        system_tokens = self._estimate_tokens(self.system_message if self.system_message else "")
        # Reserve buffer for chat template formatting (<|im_start|>, roles, etc.) and safety
        safety_buffer = 1000 
        
        # Calculate effective limit for context files
        effective_max = self.max_context_tokens - system_tokens - safety_buffer
        
        logger.info(f"Context budget: {self.max_context_tokens} total - {system_tokens} system - {safety_buffer} buffer = {effective_max} available for files")
        
        # Helper to safely add content
        def try_add_section(header, content, filepath):
            nonlocal current_tokens
            # Construct the exact text that would be added
            section_text = f"{header} ({filepath})\n\n{content}\n"
            section_tokens = self._estimate_tokens(section_text)
            
            if current_tokens + section_tokens > effective_max:
                logger.warning(f"Skipping {filepath} due to context limit ({current_tokens} + {section_tokens} > {effective_max})")
                return False
                
            parts.append(f"{header} ({filepath})")
            parts.append("")
            parts.append(content)
            parts.append("")
            current_tokens += section_tokens
            logger.info(f"Including {filepath} ({len(content)} chars, ~{section_tokens} tokens)")
            return True

        # Priority 1: Include specification (critical for correct interface)
        spec_file = None
        for file_path in self.context_files.keys():
            if 'specification' in file_path.lower() or 'spec' in file_path.lower():
                if self.context_files[file_path] is not None:
                    spec_file = file_path
                    break

        if spec_file:
            try_add_section("## Context: Specification", self.context_files[spec_file], spec_file)

        # Priority 2: Include testbench (helpful for understanding test expectations)
        tb_file = None
        for file_path in self.context_files.keys():
            if ('tb' in file_path.lower() or 'test' in file_path.lower()) and file_path != spec_file:
                if self.context_files[file_path] is not None:
                    tb_file = file_path
                    break

        if tb_file:
            try_add_section("## Context: Testbench", self.context_files[tb_file], tb_file)

        # Priority 3: Include RTL files (for bug-fix tasks where model needs to see existing code)
        rtl_files_included = []
        for file_path, content in self.context_files.items():
            if file_path not in [spec_file, tb_file] and content is not None:
                if file_path.startswith('rtl/') and len(content.strip()) >= 100:
                    if try_add_section("## Context: Existing RTL", content, file_path):
                        rtl_files_included.append(file_path)

        # Priority 4: Include any other documentation
        for file_path, content in self.context_files.items():
            if file_path not in [spec_file, tb_file] + rtl_files_included and content is not None:
                if file_path.startswith('docs/') or 'readme' in file_path.lower():
                    try_add_section("## Context", content, file_path)

        enriched_prompt = "\n".join(parts)
        logger.info(f"Built enriched prompt: {len(enriched_prompt)} chars, ~{current_tokens} tokens")
        return enriched_prompt

    def _setup_workspace(self):
        """Create CVDP workspace with context files"""
        # Create unique workspace for this worker to prevent collisions in group_size > 1
        problem_workspace = os.path.join(self.workspace_dir, f"{self.problem_id}_{self.worker_id}")
        os.makedirs(problem_workspace, exist_ok=True)

        # Write context files (docs, verif, rtl templates, etc.)
        for file_path, content in self.context_files.items():
            if content is None:
                continue
            full_path = os.path.join(problem_workspace, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Write prompt.json (for reference)
        prompt_json = {"prompt": self.prompt}
        with open(os.path.join(problem_workspace, "prompt.json"), 'w') as f:
            json.dump(prompt_json, f, indent=2)

        logger.info(f"Workspace setup complete: {problem_workspace}")

        # Setup per-turn logs directory (use log_path if provided, otherwise workspace_dir)
        if self.log_path:
            # Also make logs unique per worker
            self.turn_logs_dir = os.path.join(self.log_path, "turn_logs", f"{self.problem_id}_{self.worker_id}")
        else:
            self.turn_logs_dir = os.path.join(problem_workspace, "turn_logs")
        os.makedirs(self.turn_logs_dir, exist_ok=True)
        logger.info(f"Turn logs directory: {self.turn_logs_dir}")

        # Setup per-task logger with its own file handler
        self._setup_task_logger()

    def _setup_task_logger(self):
        """
        Setup a per-task logger that writes to a separate log file.
        This keeps each task's logs separate from others.
        """
        # Create a unique logger for this task
        self.task_logger = logging.getLogger(f"cvdp_task.{self.problem_id}.{self.worker_id}")
        self.task_logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicates
        self.task_logger.handlers = []

        # Create file handler for task-specific log
        task_log_file = os.path.join(self.turn_logs_dir, "task.log")
        file_handler = logging.FileHandler(task_log_file, mode='w')
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Add handler to task logger
        self.task_logger.addHandler(file_handler)

        # Prevent propagation to root logger (avoid duplicate logs)
        self.task_logger.propagate = False

        self.task_logger.info(f"Task logger initialized for {self.problem_id}")

    def _write_turn_log(self, turn: int, section: str, content: str):
        """
        Write content to a per-turn log file.

        Args:
            turn: Turn number (0 for initial, 1+ for subsequent turns)
            section: Section name (e.g., "INPUT_PROMPT", "MODEL_OUTPUT", "COMMAND_RESULT")
            content: Content to write
        """
        if not self.turn_logs_dir:
            return

        log_file = os.path.join(self.turn_logs_dir, f"turn_{turn}.log")
        with open(log_file, 'a') as f:
            f.write("=" * 100 + "\n")
            f.write(f"[{section}]\n")
            f.write("=" * 100 + "\n")
            f.write(content + "\n")
            f.write("\n")

    def _write_episode_summary(self, reason: str, eval_result: Dict, reward: float):
        """
        Write an episode summary file with overview of all turns.
        """
        if not self.turn_logs_dir:
            return

        summary_file = os.path.join(self.turn_logs_dir, "episode_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("EPISODE SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Problem ID: {self.problem_id}\n")
            f.write(f"Total Turns: {self.current_turn}/{self.max_turns}\n")
            f.write(f"End Reason: {reason}\n\n")

            f.write("EVALUATION RESULTS:\n")
            f.write(f"  Format Valid: {eval_result['format_valid']}\n")
            f.write(f"  Syntax Valid: {eval_result['syntax_valid']}\n")
            f.write(f"  Tests Passed: {eval_result['tests_passed']}\n")
            f.write(f"  Pass Rate: {eval_result.get('pass_rate', 0.0):.2%}\n")
            f.write(f"  Final Reward: {reward:.4f}\n\n")

            f.write("TURN LOG FILES:\n")
            for i in range(self.current_turn + 1):
                turn_file = f"turn_{i}.log"
                if os.path.exists(os.path.join(self.turn_logs_dir, turn_file)):
                    f.write(f"  - {turn_file}\n")

            f.write("\n" + "=" * 100 + "\n")

    async def _start_docker_container(self) -> str:
        """
        Start persistent Docker container for this episode.

        Returns:
            Container ID
        """
        problem_workspace = os.path.join(self.workspace_dir, f"{self.problem_id}_{self.worker_id}")
        abs_workspace = os.path.abspath(problem_workspace)

        # Start container with workspace mounted at /code
        cmd = [
            'docker', 'run',
            '-d',  # Detached mode
            '-i',  # Keep STDIN open
            '--rm',  # Auto-remove when stopped
            '--user', f'{os.getuid()}:{os.getgid()}',  # Run as current user to avoid root ownership issues
            '-v', f'{abs_workspace}:/code',  # Mount workspace
            '-w', '/code',  # Working directory
            self.docker_image,
            'bash',  # Keep container running
        ]

        self.task_logger.info(f"Starting Docker container: {self.docker_image}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace')
            self.task_logger.error(f"Failed to start Docker container: {error_msg}")
            raise RuntimeError(f"Docker container start failed: {error_msg}")

        container_id = stdout.decode('utf-8').strip()
        self.task_logger.info(f"Docker container started: {container_id[:12]}")

        return container_id

    async def _execute_command_in_container(self, command: str) -> tuple[str, str, int]:
        """
        Execute bash command in Docker container.

        Args:
            command: Bash command to execute

        Returns:
            (stdout, stderr, returncode)
        """
        if not self.docker_container_id:
            raise RuntimeError("Docker container not started")

        # Execute command in container
        cmd = [
            'docker', 'exec',
            '-i',  # Interactive
            '-w', '/code',  # Working directory
            self.docker_container_id,
            'bash', '-c', command
        ]

        self.task_logger.info(f"Executing command in container: {command}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )

            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            returncode = proc.returncode

            self.task_logger.info(f"Command completed with exit code {returncode}")

            return stdout_str, stderr_str, returncode

        except asyncio.TimeoutError:
            self.task_logger.warning(f"Command timeout: {command[:100]}")
            return "", "Command timed out", 124

    async def _stop_docker_container(self):
        """Stop and cleanup Docker container"""
        if self.docker_container_id:
            self.task_logger.info(f"Stopping Docker container: {self.docker_container_id[:12]}")
            try:
                await asyncio.create_subprocess_exec(
                    'docker', 'stop', self.docker_container_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception as e:
                self.task_logger.error(f"Error stopping container: {e}")

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Return the RTL design task with full context and start Docker container.
        """
        # Build enriched prompt with all context
        enriched_prompt = self._build_prompt_with_context()

        # Build conversation with system message and user prompt
        messages: List[renderers.Message] = []

        # Add system message
        if self.system_message:
            messages.append(renderers.Message(role="system", content=self.system_message))

        # Add user message with task
        messages.append(renderers.Message(role="user", content=enriched_prompt))

        # Store conversation history
        self.conversation_history = messages.copy()

        # Log episode start
        self.task_logger.info(f"AGENTIC EPISODE START: {self.problem_id} (max {self.max_turns} turns)")
        self.task_logger.info(f"Turn logs: {self.turn_logs_dir}")

        # Start Docker container
        try:
            self.docker_container_id = await self._start_docker_container()
        except Exception as e:
            self.task_logger.error(f"Failed to start Docker container: {e}")
            # Continue without container - will fail gracefully in step()

        # Use standard tinker renderer to build prompt
        model_input = self.renderer.build_generation_prompt(self.conversation_history)

        # Log the input prompt to per-turn log file (turn 0 = initial)
        prompt_text = self.renderer.tokenizer.decode(list(model_input.to_ints()))
        self._write_turn_log(0, "EPISODE_INFO", f"Problem ID: {self.problem_id}\nMax turns: {self.max_turns}")
        self._write_turn_log(0, "INPUT_PROMPT", prompt_text)

        return model_input, self.stop_condition

    def _extract_commands_from_text(self, generated_text: str) -> list[dict[str, str | None]]:
        """
        Extract bash commands from the model's output text.

        Supports multiple formats:
        1. <tool_call>{"name": "execute_bash", "args": {"command": "..."}}</tool_call>
        2. ```bash ... ``` code blocks (fallback)

        Returns a list of dicts: {'command': str|None, 'error': str|None}
        Special case: Returns [{'command': None, 'error': 'TRUNCATED'}] if tool call was cut off
        """
        commands = []

        # First, check for truncated tool calls (has opening tag but no closing tag)
        has_tool_call_open = '<tool_call>' in generated_text
        has_tool_call_close = '</tool_call>' in generated_text

        if has_tool_call_open and not has_tool_call_close:
            self.task_logger.warning("[extract] Detected TRUNCATED tool_call - response was cut off")
            return [{"command": None, "error": "TRUNCATED"}]

        # Method 1: Parse <tool_call> tags (Qwen's native format)
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, generated_text, re.DOTALL)

        for match in tool_call_matches:
            try:
                tool_call = json.loads(match)
                if tool_call.get("name") == "execute_bash":
                    command = tool_call.get("args", {}).get("command")
                    if command:
                        commands.append({"command": command, "error": None})
                        self.task_logger.info(f"[extract] Found tool_call command: {command[:100]}...")
                    else:
                        commands.append({"command": None, "error": "Missing 'command' in args"})
                else:
                    # Unknown tool - still try to extract command from args
                    command = tool_call.get("args", {}).get("command")
                    if command:
                        commands.append({"command": command, "error": None})
                        self.task_logger.info(f"[extract] Found command in unknown tool: {command[:100]}...")
            except json.JSONDecodeError as e:
                self.task_logger.warning(f"[extract] Failed to parse tool_call JSON: {e}")
                commands.append({"command": None, "error": f"JSON decode failed: {e}"})

        if commands:
            return commands

        # Method 2: Fallback - parse ```bash ... ``` code blocks
        bash_pattern = r'```(?:bash|sh)?\s*\n(.*?)\n```'
        bash_matches = re.findall(bash_pattern, generated_text, re.DOTALL)

        for match in bash_matches:
            command = match.strip()
            if command:
                commands.append({"command": command, "error": None})
                self.task_logger.info(f"[extract] Found bash block command: {command[:100]}...")

        # Method 3: Look for explicit command markers
        if not commands:
            # Pattern: "Command: <command>" or "$ <command>"
            command_patterns = [
                r'Command:\s*(.+?)(?:\n|$)',
                r'^\$\s*(.+?)(?:\n|$)',
            ]
            for pattern in command_patterns:
                matches = re.findall(pattern, generated_text, re.MULTILINE)
                for match in matches:
                    command = match.strip()
                    if command and len(command) > 2:
                        commands.append({"command": command, "error": None})
                        self.task_logger.info(f"[extract] Found pattern command: {command[:100]}...")

        return commands

    async def step(self, action: Action) -> StepResult:
        """
        Execute one turn of the agentic loop.

        Args:
            action: Model generated tokens (list of token IDs)

        Returns:
            StepResult with observation, reward, done, info
        """
        self.current_turn += 1

        # Decode action to text
        if isinstance(action, list):
            generated_text = self.renderer.tokenizer.decode(action)
        else:
            generated_text = str(action)

        self.task_logger.info(f"TURN {self.current_turn}/{self.max_turns}")

        # Log model output to per-turn log file
        self._write_turn_log(self.current_turn, "MODEL_OUTPUT", generated_text)

        # Add assistant response to history
        self.conversation_history.append(
            renderers.Message(role="assistant", content=generated_text)
        )

        # Extract commands from text
        extracted_items = self._extract_commands_from_text(generated_text)

        if not extracted_items:
            # No command found - treat as final answer attempt
            self.task_logger.info("No command found in response - checking for final answer")
            return await self._handle_final_answer(generated_text)

        # Check for truncated tool call (special case)
        if len(extracted_items) == 1 and extracted_items[0].get("error") == "TRUNCATED":
            self.task_logger.warning("Tool call was truncated - giving feedback to model")
            truncation_feedback = (
                "ERROR: Your response was cut off before the </tool_call> tag. "
                "Your command was too long and got truncated.\n\n"
                "Please try again with a SHORTER command. Tips:\n"
                "- Break large file writes into multiple smaller commands\n"
                "- Instead of writing entire files, write sections incrementally\n"
                "- Use 'echo' or 'printf' for smaller file contents\n"
                "- Focus on one task at a time\n\n"
                "Try your command again, but make it shorter."
            )
            self.conversation_history.append(
                renderers.Message(role="user", content=f"Tool output:\n{truncation_feedback}")
            )
            # Check for max turns
            if self.current_turn >= self.max_turns:
                self.task_logger.warning(f"Max turns ({self.max_turns}) reached. Ending episode.")
                return await self._handle_episode_end("max_turns")

            # Truncate context if needed before retry
            self._truncate_context_if_needed()

            # Return new prompt for retry
            model_input = self.renderer.build_generation_prompt(self.conversation_history)

            # Log the truncation feedback and next input prompt
            self._write_turn_log(self.current_turn, "TRUNCATION_FEEDBACK", truncation_feedback)
            prompt_text = self.renderer.tokenizer.decode(list(model_input.to_ints()))
            self._write_turn_log(self.current_turn, "NEXT_INPUT_PROMPT", prompt_text)

            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=model_input,
                next_stop_condition=self.stop_condition,
                metrics={"status": 0}
            )

        # Execute all commands and collect results
        all_observations = []

        for i, item in enumerate(extracted_items):
            command = item.get("command")
            error = item.get("error")

            if error:
                self.task_logger.info(f"Processing parsing error {i+1}/{len(extracted_items)}: {error}")
                observation_text = f"Error parsing command: {error}"
                stdout, stderr, returncode = "", error, 1
            else:
                self.task_logger.info(f"Executing command {i+1}/{len(extracted_items)}: {command}")
                # Also log to module logger for high-level overview in main logs.log
                cmd_preview = command[:120] + "..." if len(command) > 120 else command
                logger.info(f"[{self.problem_id}] Turn {self.current_turn} CMD: {cmd_preview}")

                # Execute command in Docker container
                stdout, stderr, returncode = await self._execute_command_in_container(command)

                # Format observation
                observation_text = format_command_observation(command, stdout, stderr, returncode)

            observation_text = truncate_output(observation_text, max_chars=4000)  # Aggressive truncation for context
            all_observations.append(observation_text)

            # Log command and result to per-turn log file
            self._write_turn_log(self.current_turn, f"COMMAND_{i+1}", command if command else f"Error: {error}")
            self._write_turn_log(self.current_turn, f"COMMAND_{i+1}_RESULT", observation_text)

            # Check if episode should end (only for valid commands)
            if not error:
                should_end, end_reason = self._check_episode_done(command, stdout, stderr, returncode)

                if should_end:
                    self.task_logger.info(f"Episode ending: {end_reason}")
                    return await self._handle_episode_end(end_reason)

        # Add tool response to conversation history
        combined_observation = "\n\n".join(all_observations)
        self.conversation_history.append(
            renderers.Message(role="user", content=f"Tool output:\n{combined_observation}")
        )

        # Check for max turns
        if self.current_turn >= self.max_turns:
            logger.warning(f"Max turns ({self.max_turns}) reached. Ending episode.")
            return await self._handle_episode_end("max_turns")

        # Truncate context if needed before next turn
        self._truncate_context_if_needed()

        # Return new prompt (encoded history)
        model_input = self.renderer.build_generation_prompt(self.conversation_history)

        # Log the next input prompt to per-turn log file (for next turn)
        prompt_text = self.renderer.tokenizer.decode(list(model_input.to_ints()))
        self._write_turn_log(self.current_turn + 1, "INPUT_PROMPT", prompt_text)

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=model_input,
            next_stop_condition=self.stop_condition,
            metrics={"status": 0}  # 0 = continuing
        )

    async def _handle_final_answer(self, generated_text: str) -> StepResult:
        """
        Handle case where model provides final answer without command.
        """
        self.task_logger.info("Model provided final answer - running evaluation")
        return await self._handle_episode_end("final_answer")

    async def _handle_episode_end(self, reason: str) -> StepResult:
        """
        Handle episode end - run final CVDP evaluation.
        """
        self.episode_ended = True

        # Copy files from container to workspace before evaluation
        if self.docker_container_id:
            self.task_logger.info("Copying files from container to workspace...")
            try:
                # Copy /code/. to workspace_dir/problem_id_workerid/
                dest_dir = os.path.join(self.workspace_dir, f"{self.problem_id}_{self.worker_id}")
                proc = await asyncio.create_subprocess_exec(
                    'docker', 'cp', f'{self.docker_container_id}:/code/.', dest_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    self.task_logger.error(f"Failed to copy files: {stderr.decode()}")
                else:
                    self.task_logger.info("Files copied successfully.")
            except Exception as e:
                self.task_logger.error(f"Error copying files: {e}")

        # Run CVDP evaluation
        eval_result = await self._run_cvdp_evaluation()

        # Calculate reward
        reward = self._calculate_reward(eval_result)

        # Log results
        episode_summary = (
            f"Reason: {reason}\n"
            f"Turns used: {self.current_turn}/{self.max_turns}\n"
            f"Format Valid: {eval_result['format_valid']}\n"
            f"Syntax Valid: {eval_result['syntax_valid']}\n"
            f"Tests Passed: {eval_result['tests_passed']}\n"
            f"Pass Rate: {eval_result.get('pass_rate', 0.0):.2%}\n"
            f"Final Reward: {reward:.4f}"
        )
        self.task_logger.info(f"EPISODE END: {reason}, reward={reward:.4f}")
        self._write_turn_log(self.current_turn, "EPISODE_END", episode_summary)

        # Also log episode summary to main logs.log for easy review of all tasks
        logger.info(f"EPISODE END [{self.problem_id}, {self.difficulty}]: {reason}, turns={self.current_turn}/{self.max_turns}, "
                    f"format={eval_result['format_valid']}, syntax={eval_result['syntax_valid']}, "
                    f"tests={eval_result['tests_passed']}, pass_rate={eval_result.get('pass_rate', 0.0):.2%}, "
                    f"reward={reward:.4f}")

        # Write episode summary file
        self._write_episode_summary(reason, eval_result, reward)

        # Stop Docker container
        await self._stop_docker_container()

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format_valid": float(eval_result["format_valid"]),
                "syntax_valid": float(eval_result["syntax_valid"]),
                "tests_passed": float(eval_result["tests_passed"]),
                "pass_rate": eval_result.get("pass_rate", 0.0),
                "turns_used": self.current_turn,
            }
        )

    def _check_episode_done(self, command: str, stdout: str, stderr: str, returncode: int) -> tuple[bool, str]:
        """
        Check if episode should end based on command output.

        Episode ends if:
        1. Tests passed (pytest shows all tests passed)
        2. Max turns reached (checked in step)

        Args:
            command: Command that was executed
            stdout: Command stdout
            stderr: Command stderr
            returncode: Command exit code

        Returns:
            (should_end: bool, reason: str)
        """
        # Check for pytest success
        if 'pytest' in command.lower():
            # Look for "N passed" with no failures
            passed_match = re.search(r'(\d+)\s+passed', stdout)
            failed_match = re.search(r'(\d+)\s+failed', stdout)

            if passed_match and not failed_match:
                num_passed = int(passed_match.group(1))
                if num_passed > 0 and returncode == 0:
                    return True, "tests_passed"

        # Check for explicit success markers
        if "all tests passed" in stdout.lower() or "all tests passed" in stderr.lower():
            return True, "tests_passed"

        # Continue episode
        return False, ""

    async def _run_cvdp_evaluation(self) -> Dict:
        """
        Run CVDP harness using Docker Compose for final evaluation.

        Returns:
            Dict with evaluation results
        """
        problem_workspace = os.path.join(self.workspace_dir, f"{self.problem_id}_{self.worker_id}")
        harness_dir = os.path.join(problem_workspace, "harness", "1")
        os.makedirs(harness_dir, exist_ok=True)

        service_name = "direct"  # Default service name

        # Write harness files
        for file_path, content in self.harness_config.items():
            if content is None:
                continue

            full_path = os.path.join(harness_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Replace __OSS_SIM_IMAGE__ placeholder
            if "docker-compose.yml" in file_path:
                content = content.replace("__OSS_SIM_IMAGE__", "ghcr.io/hdl/sim/osvb:latest")
                # Fix: Add missing volume mount for code directory
                if "- ./code:/code" not in content:
                    content = content.replace("- ./src/:/src/:ro", "- ./src/:/src/:ro\n      - ./code:/code")

                # Detect service name from docker-compose.yml
                match = re.search(r'services:\s*\n\s+([a-zA-Z0-9_\-]+):', content)
                if match:
                    service_name = match.group(1)
                    self.task_logger.info(f"Detected docker service name: {service_name}")

            with open(full_path, 'w') as f:
                f.write(content)

        # Create code symlink
        code_link = os.path.join(harness_dir, "code")
        if not os.path.exists(code_link):
            try:
                abs_problem_workspace = os.path.abspath(problem_workspace)
                os.symlink(abs_problem_workspace, code_link, target_is_directory=True)
            except FileExistsError:
                pass

        # Run Docker Compose
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'compose', '-f', 'docker-compose.yml', 'up',
                '--abort-on-container-exit', '--exit-code-from', service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=harness_dir
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds
            )

            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')

            # Log evaluation output to file
            try:
                eval_log_path = os.path.join(self.turn_logs_dir, "eval_pipeline_output.log")
                with open(eval_log_path, "w") as f:
                    f.write("STDOUT:\n")
                    f.write(stdout_str)
                    f.write("\n\nSTDERR:\n")
                    f.write(stderr_str)
            except Exception as e:
                self.task_logger.error(f"Failed to write eval log: {e}")
            returncode = proc.returncode

        except asyncio.TimeoutError:
            self.task_logger.warning(f"Docker evaluation timeout for {self.problem_id}")
            return {
                "format_valid": True,
                "syntax_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
            }
        except Exception as e:
            self.task_logger.error(f"Docker evaluation error: {e}")
            return {
                "format_valid": True,
                "syntax_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
            }
        finally:
            # Cleanup Docker containers
            await asyncio.create_subprocess_exec(
                'docker', 'compose', '-f', 'docker-compose.yml', 'down',
                cwd=harness_dir,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        # Parse results
        return self._parse_harness_output(stdout_str, stderr_str, returncode)

    def _parse_harness_output(self, stdout: str, stderr: str, returncode: int) -> Dict:
        """Parse cocotb test results from Docker output"""
        # Check syntax validity
        syntax_errors = [
            "SyntaxError",
            "ParseError",
            "compilation failed",
            "Compilation error",
        ]
        syntax_valid = not any(err in stderr or err in stdout for err in syntax_errors)

        # Check test results
        passed_pattern = r"(\d+)\s+passed"
        failed_pattern = r"(\d+)\s+failed"

        passed_match = re.search(passed_pattern, stdout, re.IGNORECASE)
        failed_match = re.search(failed_pattern, stdout, re.IGNORECASE)

        num_passed = int(passed_match.group(1)) if passed_match else 0
        num_failed = int(failed_match.group(1)) if failed_match else 0

        total_tests = num_passed + num_failed
        tests_passed = (num_failed == 0 and num_passed > 0) or returncode == 0
        pass_rate = num_passed / total_tests if total_tests > 0 else 0.0

        return {
            "format_valid": True,  # Agentic mode doesn't require format check
            "syntax_valid": syntax_valid,
            "tests_passed": tests_passed,
            "pass_rate": pass_rate,
        }

    def _calculate_reward(self, eval_result: Dict) -> float:
        """
        Calculate reward based on evaluation results.

        Reward structure:
        - Format valid: format_coef (default 0.1)
        - Syntax valid: syntax_coef (default 0.3)
        - Tests passed: test_coef * pass_rate (default 1.0)

        Total maximum reward: 1.4
        """
        reward = 0.0

        if eval_result["format_valid"]:
            reward += self.format_coef

        if eval_result["syntax_valid"]:
            reward += self.syntax_coef

        # Use pass_rate instead of binary tests_passed for finer granularity
        reward += self.test_coef * eval_result.get("pass_rate", 0.0)

        return reward

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup container"""
        await self._stop_docker_container()
