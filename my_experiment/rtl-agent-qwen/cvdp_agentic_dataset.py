"""
CVDP Agentic Dataset Integration for Tinker

Loads CVDP JSONL format and creates agentic multi-turn environments.
"""

import json
import logging
import math
import os
from functools import partial
from typing import List, Sequence, Tuple

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cvdp_agentic_env import CVDPAgenticEnv

logger = logging.getLogger(__name__)


class CVDPAgenticDataset(RLDataset):
    """
    Dataset that provides CVDP problems as multi-turn agentic Tinker environments.

    Each batch contains `batch_size` problems, and each problem is repeated
    `group_size` times for multiple rollouts (GRPO-style).
    """

    def __init__(
        self,
        problems: List[dict],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        workspace_dir: str,
        docker_image: str = "ghcr.io/peter-shen-simpleai/gpt-oss-20b-agent-base:latest",
        timeout_seconds: int = 600,
        max_turns: int = 50,
        format_coef: float = 0.1,
        syntax_coef: float = 0.3,
        test_coef: float = 1.0,
        dataset_name: str = "cvdp_agentic",
        override_system_message: str | None = None,
    ):
        """
        Args:
            problems: List of CVDP problem dicts (from JSONL)
            batch_size: Number of problems per batch
            group_size: Number of rollouts per problem
            renderer: Tinker renderer for tokenization
            workspace_dir: Directory to store CVDP workspaces
            docker_image: Docker image with HDL tools (iverilog, pytest, etc.)
            timeout_seconds: Command execution timeout
            max_turns: Maximum turns per episode
            format_coef: Reward coefficient for valid format
            syntax_coef: Reward coefficient for syntax validity
            test_coef: Reward coefficient for passing tests
            dataset_name: Name for logging/metrics
            override_system_message: If provided, use this instead of JSONL system_message
        """
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.workspace_dir = workspace_dir
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns
        self.format_coef = format_coef
        self.syntax_coef = syntax_coef
        self.test_coef = test_coef
        self.dataset_name = dataset_name
        self.override_system_message = override_system_message

        logger.info(
            f"Loaded CVDP agentic dataset: {len(problems)} problems, "
            f"batch_size={batch_size}, group_size={group_size}, max_turns={max_turns}"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """
        Get a batch of environment group builders.

        Returns a list of ProblemGroupBuilder, each creating `group_size` copies
        of the same CVDP problem.
        """
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.problems))

        if batch_start >= len(self.problems):
            raise IndexError(f"Batch index {index} out of range")

        env_group_builders = []

        for problem in self.problems[batch_start:batch_end]:
            # Use override system message if provided, otherwise use JSONL's system message
            system_message = self.override_system_message if self.override_system_message is not None else problem.get("system_message")

            # Create environment builder for this problem
            env_builder = ProblemGroupBuilder(
                env_thunk=partial(
                    CVDPAgenticEnv,
                    problem_id=problem["id"],
                    prompt=problem["prompt"],
                    context_files=problem.get("context", {}),
                    harness_config=problem.get("harness", {}),
                    system_message=system_message,
                    workspace_dir=self.workspace_dir,
                    renderer=self.renderer,
                    docker_image=self.docker_image,
                    timeout_seconds=self.timeout_seconds,
                    max_turns=self.max_turns,
                    format_coef=self.format_coef,
                    syntax_coef=self.syntax_coef,
                    test_coef=self.test_coef,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            env_group_builders.append(env_builder)

        return env_group_builders

    def __len__(self) -> int:
        """Number of batches in the dataset"""
        return math.ceil(len(self.problems) / self.batch_size)


@chz.chz
class CVDPAgenticDatasetBuilder(RLDatasetBuilder):
    """
    Builder for CVDP agentic datasets.

    Loads problems from JSONL file and creates train/test datasets.
    """

    # Required parameters
    cvdp_jsonl_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str

    # Optional parameters
    workspace_dir: str = "/tmp/cvdp_agentic_workspace"
    docker_image: str = "ghcr.io/peter-shen-simpleai/gpt-oss-20b-agent-base:latest"
    timeout_seconds: int = 600
    max_turns: int = 50
    format_coef: float = 0.1
    syntax_coef: float = 0.3
    test_coef: float = 1.0
    dataset_name: str = "cvdp_agentic"
    override_system_message: str | None = None  # Override JSONL system_message with this

    # Train/test split
    test_split: float = 0.0  # Fraction of problems for test set (0.0 = no test set)
    test_jsonl_path: str | None = None  # Optional separate test set

    async def __call__(self) -> Tuple[CVDPAgenticDataset, CVDPAgenticDataset | None]:
        """
        Build train and optionally test datasets.

        Returns:
            (train_dataset, test_dataset)
            test_dataset is None if no test split configured
        """
        # Load tokenizer and renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)

        # Load problems from JSONL
        problems = self._load_jsonl(self.cvdp_jsonl_path)
        logger.info(f"Loaded {len(problems)} problems from {self.cvdp_jsonl_path}")

        # Split train/test
        if self.test_jsonl_path:
            # Use separate test file
            train_problems = problems
            test_problems = self._load_jsonl(self.test_jsonl_path)
            logger.info(f"Loaded {len(test_problems)} test problems from {self.test_jsonl_path}")
        elif self.test_split > 0:
            # Split from same file
            split_idx = int(len(problems) * (1 - self.test_split))
            train_problems = problems[:split_idx]
            test_problems = problems[split_idx:]
            logger.info(f"Split: {len(train_problems)} train, {len(test_problems)} test")
        else:
            # No test set
            train_problems = problems
            test_problems = None

        # Create train dataset
        train_dataset = CVDPAgenticDataset(
            problems=train_problems,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            workspace_dir=os.path.join(self.workspace_dir, "train"),
            docker_image=self.docker_image,
            timeout_seconds=self.timeout_seconds,
            max_turns=self.max_turns,
            format_coef=self.format_coef,
            syntax_coef=self.syntax_coef,
            test_coef=self.test_coef,
            dataset_name=f"{self.dataset_name}_train",
            override_system_message=self.override_system_message,
        )

        # Create test dataset
        test_dataset = None
        if test_problems:
            test_dataset = CVDPAgenticDataset(
                problems=test_problems,
                batch_size=self.batch_size,
                group_size=1,  # Single rollout for evaluation
                renderer=renderer,
                workspace_dir=os.path.join(self.workspace_dir, "test"),
                docker_image=self.docker_image,
                timeout_seconds=self.timeout_seconds,
                max_turns=self.max_turns,
                format_coef=self.format_coef,
                syntax_coef=self.syntax_coef,
                test_coef=self.test_coef,
                dataset_name=f"{self.dataset_name}_test",
                override_system_message=self.override_system_message,
            )

        return train_dataset, test_dataset

    def _load_jsonl(self, jsonl_path: str) -> List[dict]:
        """Load CVDP problems from JSONL file"""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"CVDP JSONL not found: {jsonl_path}")

        problems = []
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    problem = json.loads(line)

                    # Validate required fields
                    required_fields = ["id", "prompt"]
                    for field in required_fields:
                        if field not in problem:
                            logger.warning(
                                f"Line {line_num}: Missing required field '{field}', skipping"
                            )
                            continue

                    problems.append(problem)

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
                    continue

        if not problems:
            raise ValueError(f"No valid problems found in {jsonl_path}")

        return problems
