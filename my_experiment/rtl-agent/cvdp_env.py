"""
CVDP Environment Wrapper for Tinker

Wraps the CVDP benchmark's Docker-based evaluation as a Tinker Env class.
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, Optional

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import Action, Env, Observation, StepResult
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


class CVDPEnv(Env):
    """
    Environment that evaluates RTL code using CVDP's Docker harness.

    Workflow:
    1. initial_observation() returns the RTL design task prompt
    2. Model generates Verilog/SystemVerilog code
    3. step() extracts code, runs CVDP Docker evaluation, returns reward
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
        oss_sim_image: str = "ghcr.io/hdl/sim/osvb:latest",
        timeout_seconds: int = 300,
        format_coef: float = 0.1,
        syntax_coef: float = 0.3,
        test_coef: float = 1.0,
    ):
        """
        Args:
            problem_id: Unique problem identifier (e.g., "cvdp_agentic_fixed_arbiter_0001")
            prompt: RTL design task description
            context_files: Dict of file_path -> content (docs, verif, etc.)
            harness_config: Dict of harness file_path -> content
            workspace_dir: Directory to store CVDP workspace
            renderer: Tinker renderer for tokenization
            system_message: Optional system message (instructions about file operations, etc.)
            oss_sim_image: Docker image for simulation (default: osvb)
            timeout_seconds: Docker execution timeout
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
        self.oss_sim_image = oss_sim_image
        self.timeout_seconds = timeout_seconds

        # Reward coefficients
        self.format_coef = format_coef
        self.syntax_coef = syntax_coef
        self.test_coef = test_coef

        # Setup workspace
        self._setup_workspace()

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _build_prompt_with_context(self) -> str:
        """
        Build enriched prompt including specification and relevant context files.

        This ensures the model has access to all necessary information:
        - Original task prompt
        - Specification file (critical for correct interface)
        - Testbench (helpful for understanding requirements)
        - Any other documentation

        Returns:
            Complete prompt with embedded context files
        """
        parts = [self.prompt]
        parts.append("")  # Blank line separator

        # Priority 1: Include specification (critical for correct interface)
        spec_file = None
        for file_path in self.context_files.keys():
            if 'specification' in file_path.lower() or 'spec' in file_path.lower():
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
                tb_file = file_path
                break

        if tb_file:
            parts.append(f"## Context: Testbench ({tb_file})")
            parts.append("")
            parts.append(self.context_files[tb_file])
            parts.append("")
            logger.info(f"Including testbench: {tb_file} ({len(self.context_files[tb_file])} chars)")

        # Priority 3: Include any other documentation
        for file_path, content in self.context_files.items():
            if file_path not in [spec_file, tb_file]:
                # Skip RTL template files (usually empty placeholders)
                if file_path.startswith('rtl/') and len(content.strip()) < 100:
                    continue

                # Include other documentation
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
            full_path = os.path.join(problem_workspace, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        # Write prompt.json (for agent interface)
        prompt_json = {"prompt": self.prompt}
        with open(os.path.join(problem_workspace, "prompt.json"), 'w') as f:
            json.dump(prompt_json, f, indent=2)

        logger.info(f"Workspace setup complete: {problem_workspace}")

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Return the RTL design task with full context.

        Builds enriched prompt including:
        - System message (if available)
        - Task description
        - Specification file
        - Testbench
        - Other documentation
        """
        # Build enriched prompt with all context
        enriched_prompt = self._build_prompt_with_context()

        convo = []

        # Add system message if available
        if self.system_message:
            convo.append({"role": "system", "content": self.system_message})
            logger.info(f"Including system message ({len(self.system_message)} chars)")

        # Add enriched user prompt
        convo.append({"role": "user", "content": enriched_prompt})

        logtree.log_text(f"Problem: {self.problem_id}")
        logtree.log_text(f"Total prompt length: {len(enriched_prompt)} chars")
        logtree.log_text(f"Context files included: {len(self.context_files)}")

        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Takes generated tokens, extracts Verilog code, runs CVDP evaluation, returns reward.

        Args:
            action: List of token IDs generated by the model

        Returns:
            StepResult with reward and metrics
        """
        # 1. Parse the model's response
        message, parse_success = self.renderer.parse_response(action)
        generated_text = message["content"]

        logtree.log_text(f"Generated response length: {len(generated_text)} chars")

        # 2. Extract Verilog/SystemVerilog code
        verilog_code = self._extract_verilog(generated_text)

        if verilog_code is None:
            logtree.log_text("❌ Format Error: No Verilog code block found")
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "format_valid": 0.0,
                    "syntax_valid": 0.0,
                    "tests_passed": 0.0,
                    "pass_rate": 0.0,
                }
            )

        logtree.log_text(f"✓ Extracted Verilog code: {len(verilog_code)} chars")

        # 3. Write generated code to workspace
        rtl_file_path = self._get_rtl_target_path()
        problem_workspace = os.path.join(self.workspace_dir, self.problem_id)
        full_rtl_path = os.path.join(problem_workspace, rtl_file_path)

        os.makedirs(os.path.dirname(full_rtl_path), exist_ok=True)
        with open(full_rtl_path, 'w') as f:
            f.write(verilog_code)

        logtree.log_text(f"Wrote code to: {rtl_file_path}")

        # 4. Run CVDP evaluation
        eval_result = await self._run_cvdp_evaluation(problem_workspace)

        # 5. Calculate reward
        reward = self._calculate_reward(eval_result)

        # Log results
        logtree.log_text(
            f"Evaluation: format={eval_result['format_valid']}, "
            f"syntax={eval_result['syntax_valid']}, "
            f"tests_passed={eval_result['tests_passed']}, "
            f"reward={reward:.2f}"
        )

        return StepResult(
            reward=reward,
            episode_done=True,  # Single-turn environment
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format_valid": float(eval_result["format_valid"]),
                "syntax_valid": float(eval_result["syntax_valid"]),
                "tests_passed": float(eval_result["tests_passed"]),
                "pass_rate": eval_result.get("pass_rate", 0.0),
            }
        )

    async def _run_cvdp_evaluation(self, problem_workspace: str) -> Dict:
        """
        Run CVDP harness using Docker Compose.

        Returns:
            Dict with evaluation results:
            - format_valid: bool
            - syntax_valid: bool
            - tests_passed: bool
            - pass_rate: float
            - stdout: str
            - stderr: str
        """
        harness_dir = os.path.join(problem_workspace, "harness", "1")
        os.makedirs(harness_dir, exist_ok=True)

        # Write harness files
        for file_path, content in self.harness_config.items():
            full_path = os.path.join(harness_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Replace __OSS_SIM_IMAGE__ placeholder with actual image
            if "docker-compose.yml" in file_path:
                content = content.replace("__OSS_SIM_IMAGE__", self.oss_sim_image)

            with open(full_path, 'w') as f:
                f.write(content)

        # Create code symlink for harness to access RTL/verif files
        code_link = os.path.join(harness_dir, "code")
        if not os.path.exists(code_link):
            try:
                # Use absolute path for symlink to ensure it works from harness directory
                abs_problem_workspace = os.path.abspath(problem_workspace)
                os.symlink(abs_problem_workspace, code_link, target_is_directory=True)
            except FileExistsError:
                # Another parallel rollout already created it, that's fine
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
            logger.warning(f"Docker execution timeout for {self.problem_id}")
            return {
                "format_valid": True,
                "syntax_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
                "stdout": "",
                "stderr": "Timeout",
            }
        except Exception as e:
            logger.error(f"Docker execution error for {self.problem_id}: {e}")
            return {
                "format_valid": True,
                "syntax_valid": False,
                "tests_passed": False,
                "pass_rate": 0.0,
                "stdout": "",
                "stderr": str(e),
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
        """
        Parse cocotb test results from Docker output.

        Looks for patterns like:
        - "Test Case X Passed"
        - "Test Case X Failed"
        - "X passed"
        - "ERROR"
        - "FAILED"
        """
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

        # Alternative: Look for "Test Case X Passed/Failed"
        if num_passed == 0 and num_failed == 0:
            passed_cases = len(re.findall(r"Test Case \d+ Passed", stdout))
            failed_cases = len(re.findall(r"ERROR: Test Case \d+ Failed", stdout))
            num_passed = passed_cases
            num_failed = failed_cases

        total_tests = num_passed + num_failed
        tests_passed = (num_failed == 0 and num_passed > 0) or returncode == 0
        pass_rate = num_passed / total_tests if total_tests > 0 else 0.0

        return {
            "format_valid": True,  # Already checked in step()
            "syntax_valid": syntax_valid,
            "tests_passed": tests_passed,
            "pass_rate": pass_rate,
            "num_passed": num_passed,
            "num_failed": num_failed,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _calculate_reward(self, eval_result: Dict) -> float:
        """
        Calculate reward based on evaluation results.

        Reward structure:
        - Format valid: format_coef (default 0.1)
        - Syntax valid: syntax_coef (default 0.3)
        - Tests passed: test_coef (default 1.0)

        Total maximum reward: 1.4
        """
        reward = 0.0

        if eval_result["format_valid"]:
            reward += self.format_coef

        if eval_result["syntax_valid"]:
            reward += self.syntax_coef

        if eval_result["tests_passed"]:
            reward += self.test_coef

        return reward

    def _extract_verilog(self, text: str) -> Optional[str]:
        """
        Extract Verilog/SystemVerilog code from markdown code block or agentic shell commands.

        Looks for patterns like:
        - ```verilog
        - ```systemverilog
        - ```sv
        - ```v
        - echo 'module ...' > file.v (agentic format)
        """
        # First try standard markdown code blocks
        patterns = [
            r'```(?:system)?verilog\n(.*?)```',
            r'```sv\n(.*?)```',
            r'```v\n(.*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                logger.debug(f"Extracted code using pattern: {pattern}")
                return code

        # Try agentic format: extract code from echo commands or heredocs
        # Pattern 1: echo 'module ...' > file.v or echo "module ..." > file.v
        echo_patterns = [
            r"echo\s+['\"](.+?module.+?endmodule.*?)['\"]",  # Single/double quotes
            r"echo\s+'([^']+module[^']+endmodule[^']*)'",     # Single quotes only
            r'echo\s+"([^"]+module[^"]+endmodule[^"]*)"',     # Double quotes only
        ]

        for pattern in echo_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                # Unescape newlines (\n -> actual newlines)
                code = code.replace('\\n', '\n')
                # Remove any remaining quotes or escape characters
                code = code.replace("\\'", "'").replace('\\"', '"')
                logger.info(f"Extracted code from agentic echo command ({len(code)} chars)")
                return code

        # Pattern 2: cat > file << 'EOF' ... EOF (heredoc format)
        heredoc_pattern = r"cat\s*>\s*\S+\s*<<\s*['\"]?EOF['\"]?\s*\n(.+?)\nEOF"
        match = re.search(heredoc_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Check if it contains module...endmodule
            if re.search(r'module\s+\w+.*?endmodule', code, re.DOTALL | re.IGNORECASE):
                logger.info(f"Extracted code from agentic heredoc ({len(code)} chars)")
                return code

        logger.warning("No Verilog code block found in response")
        return None

    def _get_rtl_target_path(self) -> str:
        """
        Determine the RTL file path to write generated code.

        Strategies:
        1. Parse from prompt (e.g., "...at the location:rtl/fixed_priority_arbiter.v")
        2. Look for existing RTL files in context (check harness config for expected filename)
        3. Use default "rtl/design.sv"
        """
        # Strategy 1: Check harness config for expected RTL file
        # The harness .env file specifies VERILOG_SOURCES which is the authoritative source
        for file_path, content in self.harness_config.items():
            if file_path.endswith('.env'):
                # Parse VERILOG_SOURCES from .env file
                for line in content.split('\n'):
                    if line.startswith('VERILOG_SOURCES'):
                        # Format: VERILOG_SOURCES = /code/rtl/fixed_priority_arbiter.sv
                        parts = line.split('=')
                        if len(parts) == 2:
                            full_path = parts[1].strip()
                            # Remove /code/ prefix if present
                            if full_path.startswith('/code/'):
                                path = full_path[6:]  # Remove '/code/'
                            else:
                                path = full_path
                            logger.debug(f"Found RTL path in harness config: {path}")
                            return path

        # Strategy 2: Parse from prompt (fallback)
        location_match = re.search(r"at the location:(\S+)", self.prompt)
        if location_match:
            path = location_match.group(1).strip()
            # Remove trailing punctuation (period, comma, etc.)
            path = path.rstrip('.,;:')
            logger.debug(f"Found RTL path in prompt: {path}")
            return path

        # Strategy 2: Look for RTL files in context
        for file_path in self.context_files.keys():
            if file_path.startswith("rtl/") and file_path.endswith((".v", ".sv")):
                # If it's empty or a template, use it
                content = self.context_files[file_path]
                if not content or len(content) < 100:
                    logger.debug(f"Using existing RTL template: {file_path}")
                    return file_path

        # Strategy 3: Default
        logger.debug("Using default RTL path: rtl/design.sv")
        return "rtl/design.sv"
