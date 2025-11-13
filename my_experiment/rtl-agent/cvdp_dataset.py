"""
CVDP Dataset Integration for Tinker

Loads CVDP JSONL format and creates Tinker RLDataset/RLDatasetBuilder.
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

from cvdp_env import CVDPEnv

logger = logging.getLogger(__name__)


class CVDPDataset(RLDataset):
    """
    Dataset that provides CVDP problems as Tinker environments.

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
        oss_sim_image: str = "ghcr.io/hdl/sim/osv:latest",
        timeout_seconds: int = 300,
        format_coef: float = 0.1,
        syntax_coef: float = 0.3,
        test_coef: float = 1.0,
        dataset_name: str = "cvdp",
    ):
        """
        Args:
            problems: List of CVDP problem dicts (from JSONL)
            batch_size: Number of problems per batch
            group_size: Number of rollouts per problem
            renderer: Tinker renderer for tokenization
            workspace_dir: Directory to store CVDP workspaces
            oss_sim_image: Docker image for simulation
            timeout_seconds: Docker execution timeout
            format_coef: Reward coefficient for valid format
            syntax_coef: Reward coefficient for syntax validity
            test_coef: Reward coefficient for passing tests
            dataset_name: Name for logging/metrics
        """
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.workspace_dir = workspace_dir
        self.oss_sim_image = oss_sim_image
        self.timeout_seconds = timeout_seconds
        self.format_coef = format_coef
        self.syntax_coef = syntax_coef
        self.test_coef = test_coef
        self.dataset_name = dataset_name

        logger.info(
            f"Loaded CVDP dataset: {len(problems)} problems, "
            f"batch_size={batch_size}, group_size={group_size}"
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
            # Create environment builder for this problem
            env_builder = ProblemGroupBuilder(
                env_thunk=partial(
                    CVDPEnv,
                    problem_id=problem["id"],
                    prompt=problem["prompt"],
                    context_files=problem.get("context", {}),
                    harness_config=problem.get("harness", {}),
                    workspace_dir=self.workspace_dir,
                    renderer=self.renderer,
                    oss_sim_image=self.oss_sim_image,
                    timeout_seconds=self.timeout_seconds,
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
class CVDPDatasetBuilder(RLDatasetBuilder):
    """
    Builder for CVDP datasets.

    Loads problems from JSONL file and creates train/test datasets.
    """

    # Required parameters
    cvdp_jsonl_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str

    # Optional parameters
    workspace_dir: str = "/tmp/cvdp_workspace"
    oss_sim_image: str = "ghcr.io/hdl/sim/osv:latest"
    timeout_seconds: int = 300
    format_coef: float = 0.1
    syntax_coef: float = 0.3
    test_coef: float = 1.0
    dataset_name: str = "cvdp"

    # Train/test split
    test_split: float = 0.0  # Fraction of problems for test set (0.0 = no test set)
    test_jsonl_path: str | None = None  # Optional separate test set

    async def __call__(self) -> Tuple[CVDPDataset, CVDPDataset | None]:
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
        train_dataset = CVDPDataset(
            problems=train_problems,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            workspace_dir=os.path.join(self.workspace_dir, "train"),
            oss_sim_image=self.oss_sim_image,
            timeout_seconds=self.timeout_seconds,
            format_coef=self.format_coef,
            syntax_coef=self.syntax_coef,
            test_coef=self.test_coef,
            dataset_name=f"{self.dataset_name}_train",
        )

        # Create test dataset
        test_dataset = None
        if test_problems:
            test_dataset = CVDPDataset(
                problems=test_problems,
                batch_size=self.batch_size,
                group_size=1,  # Single rollout for evaluation
                renderer=renderer,
                workspace_dir=os.path.join(self.workspace_dir, "test"),
                oss_sim_image=self.oss_sim_image,
                timeout_seconds=self.timeout_seconds,
                format_coef=self.format_coef,
                syntax_coef=self.syntax_coef,
                test_coef=self.test_coef,
                dataset_name=f"{self.dataset_name}_test",
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


def load_cvdp_dataset_from_dir(
    dataset_dir: str,
    batch_size: int = 8,
    group_size: int = 4,
    model_name: str = "meta-llama/Llama-3.2-1B",
    renderer_name: str = "role_colon",
) -> Tuple[CVDPDataset, CVDPDataset | None]:
    """
    Helper function to load CVDP dataset from directory structure.

    Expects directory with:
    - train.jsonl
    - test.jsonl (optional)

    Args:
        dataset_dir: Directory containing JSONL files
        batch_size: Number of problems per batch
        group_size: Number of rollouts per problem
        model_name: Model name for tokenizer
        renderer_name: Renderer name

    Returns:
        (train_dataset, test_dataset)
    """
    train_path = os.path.join(dataset_dir, "train.jsonl")
    test_path = os.path.join(dataset_dir, "test.jsonl")

    builder = CVDPDatasetBuilder(
        cvdp_jsonl_path=train_path,
        batch_size=batch_size,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        test_jsonl_path=test_path if os.path.exists(test_path) else None,
    )

    import asyncio
    return asyncio.run(builder())
