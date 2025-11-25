"""
CVDP Agentic Environment for Tinker - Qwen Version

Multi-turn environment where the model can execute bash commands,
read files, write RTL, compile, test, debug, and iterate until tests pass.

This version uses standard tinker renderers and text-based tool parsing,
compatible with Qwen models.
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Optional, List

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
        timeout_seconds: int = 600,
        max_turns: int = 50,
        format_coef: float = 0.1,
        syntax_coef: float = 0.3,
        test_coef: float = 1.0,
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
        """
        self.problem_id = problem_id
        self.prompt = prompt
        self.context_files = context_files
        self.harness_config = harness_config
        self.system_message = system_message
        self.workspace_dir = workspace_dir
        self.renderer = renderer
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns

        # Reward coefficients
        self.format_coef = format_coef
        self.syntax_coef = syntax_coef
        self.test_coef = test_coef

        # Episode state
        self.current_turn = 0
        self.conversation_history: List[renderers.Message] = []
        self.docker_container_id: Optional[str] = None
        self.episode_ended = False

        # Setup workspace
        self._setup_workspace()

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _build_prompt_with_context(self) -> str:
        """
        Build enriched prompt including specification and relevant context files.

        Returns:
            Complete prompt with embedded context files
        """
        parts = [self.prompt]
        parts.append("")  # Blank line separator

        # Priority 1: Include specification (critical for correct interface)
        spec_file = None
        for file_path in self.context_files.keys():
            if 'specification' in file_path.lower() or 'spec' in file_path.lower():
                if self.context_files[file_path] is not None:
                    spec_file = file_path
                    break

        if spec_file:
            parts.append(f"## Context: Specification ({spec_file})")
            parts.append("")
            parts.append(self.context_files[spec_file])
            parts.append("")
            logger.info(f"Including specification: {spec_file} ({len(self.context_files[spec_file])} chars)")

        # Priority 2: Include testbench (helpful for understanding test expectations)
        tb_file = None
        for file_path in self.context_files.keys():
            if ('tb' in file_path.lower() or 'test' in file_path.lower()) and file_path != spec_file:
                if self.context_files[file_path] is not None:
                    tb_file = file_path
                    break

        if tb_file:
            parts.append(f"## Context: Testbench ({tb_file})")
            parts.append("")
            parts.append(self.context_files[tb_file])
            parts.append("")
            logger.info(f"Including testbench: {tb_file} ({len(self.context_files[tb_file])} chars)")

        # Priority 3: Include RTL files (for bug-fix tasks where model needs to see existing code)
        rtl_files_included = []
        for file_path, content in self.context_files.items():
            if file_path not in [spec_file, tb_file] and content is not None:
                if file_path.startswith('rtl/') and len(content.strip()) >= 100:
                    parts.append(f"## Context: Existing RTL ({file_path})")
                    parts.append("")
                    parts.append(content)
                    parts.append("")
                    logger.info(f"Including RTL file: {file_path} ({len(content)} chars)")
                    rtl_files_included.append(file_path)

        # Priority 4: Include any other documentation
        for file_path, content in self.context_files.items():
            if file_path not in [spec_file, tb_file] + rtl_files_included and content is not None:
                if file_path.startswith('docs/') or 'readme' in file_path.lower():
                    parts.append(f"## Context: {file_path}")
                    parts.append("")
                    parts.append(content)
                    parts.append("")
                    logger.info(f"Including documentation: {file_path} ({len(content)} chars)")

        enriched_prompt = "\n".join(parts)
        logger.info(f"Built enriched prompt: {len(enriched_prompt)} total chars")
        return enriched_prompt

    def _setup_workspace(self):
        """Create CVDP workspace with context files"""
        problem_workspace = os.path.join(self.workspace_dir, self.problem_id)
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

    async def _start_docker_container(self) -> str:
        """
        Start persistent Docker container for this episode.

        Returns:
            Container ID
        """
        problem_workspace = os.path.join(self.workspace_dir, self.problem_id)
        abs_workspace = os.path.abspath(problem_workspace)

        # Start container with workspace mounted at /code
        cmd = [
            'docker', 'run',
            '-d',  # Detached mode
            '-i',  # Keep STDIN open
            '--rm',  # Auto-remove when stopped
            '-v', f'{abs_workspace}:/code',  # Mount workspace
            '-w', '/code',  # Working directory
            self.docker_image,
            'bash',  # Keep container running
        ]

        logger.info(f"Starting Docker container: {self.docker_image}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace')
            logger.error(f"Failed to start Docker container: {error_msg}")
            raise RuntimeError(f"Docker container start failed: {error_msg}")

        container_id = stdout.decode('utf-8').strip()
        logger.info(f"Docker container started: {container_id[:12]}")

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

        logger.info(f"Executing command in container: {command[:100]}")

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

            logger.info(f"Command completed with exit code {returncode}")

            return stdout_str, stderr_str, returncode

        except asyncio.TimeoutError:
            logger.warning(f"Command timeout: {command[:100]}")
            return "", "Command timed out", 124

    async def _stop_docker_container(self):
        """Stop and cleanup Docker container"""
        if self.docker_container_id:
            logger.info(f"Stopping Docker container: {self.docker_container_id[:12]}")
            try:
                await asyncio.create_subprocess_exec(
                    'docker', 'stop', self.docker_container_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception as e:
                logger.error(f"Error stopping container: {e}")

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

        # Log the prompt
        logger.info("=" * 100)
        logger.info("AGENTIC EPISODE START (Qwen)")
        logger.info("=" * 100)
        logger.info(f"Problem ID: {self.problem_id}")
        logger.info(f"Max turns: {self.max_turns}")
        logger.info("=" * 100)

        # Start Docker container
        try:
            self.docker_container_id = await self._start_docker_container()
        except Exception as e:
            logger.error(f"Failed to start Docker container: {e}")
            # Continue without container - will fail gracefully in step()

        # Use standard tinker renderer to build prompt
        model_input = self.renderer.build_generation_prompt(self.conversation_history)
        return model_input, self.stop_condition

    def _extract_commands_from_text(self, generated_text: str) -> list[dict[str, str | None]]:
        """
        Extract bash commands from the model's output text.

        Supports multiple formats:
        1. <tool_call>{"name": "execute_bash", "args": {"command": "..."}}</tool_call>
        2. ```bash ... ``` code blocks (fallback)

        Returns a list of dicts: {'command': str|None, 'error': str|None}
        """
        commands = []

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
                        logger.info(f"[extract] Found tool_call command: {command[:100]}...")
                    else:
                        commands.append({"command": None, "error": "Missing 'command' in args"})
                else:
                    # Unknown tool - still try to extract command from args
                    command = tool_call.get("args", {}).get("command")
                    if command:
                        commands.append({"command": command, "error": None})
                        logger.info(f"[extract] Found command in unknown tool: {command[:100]}...")
            except json.JSONDecodeError as e:
                logger.warning(f"[extract] Failed to parse tool_call JSON: {e}")
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
                logger.info(f"[extract] Found bash block command: {command[:100]}...")

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
                        logger.info(f"[extract] Found pattern command: {command[:100]}...")

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

        logger.info("=" * 100)
        logger.info(f"TURN {self.current_turn}/{self.max_turns}")
        logger.info("=" * 100)
        logger.info(f"Model response ({len(generated_text)} chars):")
        logger.info("-" * 100)
        logger.info(generated_text[:2000] if len(generated_text) > 2000 else generated_text)
        logger.info("=" * 100)

        # Add assistant response to history
        self.conversation_history.append(
            renderers.Message(role="assistant", content=generated_text)
        )

        # Extract commands from text
        extracted_items = self._extract_commands_from_text(generated_text)

        if not extracted_items:
            # No command found - treat as final answer attempt
            logger.info("No command found in response - checking for final answer")
            return await self._handle_final_answer(generated_text)

        # Execute all commands and collect results
        all_observations = []

        for i, item in enumerate(extracted_items):
            command = item.get("command")
            error = item.get("error")

            if error:
                logger.info(f"Processing parsing error {i+1}/{len(extracted_items)}: {error}")
                observation_text = f"Error parsing command: {error}"
                stdout, stderr, returncode = "", error, 1
            else:
                logger.info(f"Executing command {i+1}/{len(extracted_items)}: {command}")

                # Execute command in Docker container
                stdout, stderr, returncode = await self._execute_command_in_container(command)

                # Format observation
                observation_text = format_command_observation(command, stdout, stderr, returncode)

            observation_text = truncate_output(observation_text, max_chars=10000)
            all_observations.append(observation_text)

            logger.info("Command execution result:")
            logger.info("-" * 100)
            logger.info(observation_text[:1000] if len(observation_text) > 1000 else observation_text)
            logger.info("=" * 100)

            # Check if episode should end (only for valid commands)
            if not error:
                should_end, end_reason = self._check_episode_done(command, stdout, stderr, returncode)

                if should_end:
                    logger.info(f"Episode ending: {end_reason}")
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

        # Return new prompt (encoded history)
        model_input = self.renderer.build_generation_prompt(self.conversation_history)
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
        logger.info("Model provided final answer - running evaluation")
        return await self._handle_episode_end("final_answer")

    async def _handle_episode_end(self, reason: str) -> StepResult:
        """
        Handle episode end - run final CVDP evaluation.
        """
        self.episode_ended = True

        # Copy files from container to workspace before evaluation
        if self.docker_container_id:
            logger.info("Copying files from container to workspace...")
            try:
                # Copy /code/. to workspace_dir/problem_id/
                dest_dir = os.path.join(self.workspace_dir, self.problem_id)
                proc = await asyncio.create_subprocess_exec(
                    'docker', 'cp', f'{self.docker_container_id}:/code/.', dest_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    logger.error(f"Failed to copy files: {stderr.decode()}")
                else:
                    logger.info("Files copied successfully.")
            except Exception as e:
                logger.error(f"Error copying files: {e}")

        # Run CVDP evaluation
        eval_result = await self._run_cvdp_evaluation()

        # Calculate reward
        reward = self._calculate_reward(eval_result)

        # Log results
        logger.info("=" * 100)
        logger.info("EPISODE END")
        logger.info("=" * 100)
        logger.info(f"Reason: {reason}")
        logger.info(f"Turns used: {self.current_turn}/{self.max_turns}")
        logger.info(f"Format Valid: {eval_result['format_valid']}")
        logger.info(f"Syntax Valid: {eval_result['syntax_valid']}")
        logger.info(f"Tests Passed: {eval_result['tests_passed']}")
        logger.info(f"Pass Rate: {eval_result.get('pass_rate', 0.0):.2%}")
        logger.info(f"Final Reward: {reward:.4f}")
        logger.info("=" * 100)

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
        problem_workspace = os.path.join(self.workspace_dir, self.problem_id)
        harness_dir = os.path.join(problem_workspace, "harness", "1")
        os.makedirs(harness_dir, exist_ok=True)

        # Write harness files
        for file_path, content in self.harness_config.items():
            if content is None:
                continue

            full_path = os.path.join(harness_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Replace __OSS_SIM_IMAGE__ placeholder
            if "docker-compose.yml" in file_path:
                content = content.replace("__OSS_SIM_IMAGE__", "ghcr.io/hdl/sim/osvb:latest")

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
                '--abort-on-container-exit', '--exit-code-from', 'direct',
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
            returncode = proc.returncode

        except asyncio.TimeoutError:
            logger.warning(f"Docker evaluation timeout for {self.problem_id}")
            return {
                "format_valid": True,
                "syntax_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
            }
        except Exception as e:
            logger.error(f"Docker evaluation error: {e}")
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
