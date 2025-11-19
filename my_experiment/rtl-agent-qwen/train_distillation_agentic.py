"""
Agentic On-Policy Distillation Training Script for RTL Code Generation

Multi-turn agentic training where the student model learns to:
- Execute bash commands (ls, cat, echo, iverilog, pytest)
- Read specifications and create tests
- Write RTL files iteratively
- Debug compilation/test failures
- Iterate until tests pass

Teacher model provides KL penalty on student's action tokens.
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig, TeacherConfig

from cvdp_agentic_dataset import CVDPAgenticDatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for agentic on-policy distillation"""

    # Model configuration
    student_model: str = "gpt-oss-20b"
    teacher_model: str = "Qwen/Qwen3-Coder-30B"
    teacher_checkpoint: str | None = None
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset configuration
    cvdp_jsonl_path: str = "cvdp_16_easy_problems.jsonl"
    test_jsonl_path: str | None = None
    test_split: float = 0.0

    # CVDP agentic configuration
    workspace_dir: str = "/tmp/cvdp_agentic_workspace"
    docker_image: str = "gpt-oss-20b-agent-base:latest"
    timeout_seconds: int = 600
    max_turns: int = 50

    # Reward shaping
    format_coef: float = 0.1
    syntax_coef: float = 0.3
    test_coef: float = 1.0

    # Training hyperparameters
    batch_size: int = 4  # Smaller batch size for multi-turn (more tokens per episode)
    group_size: int = 2  # Fewer rollouts for distillation
    learning_rate: float = 1e-4
    max_tokens: int = 4096  # Per turn (context accumulates across turns)

    # Distillation hyperparameters
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Advanced training options
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"
    compute_post_kl: bool = False

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def main(cli_config: CLIConfig):
    """Main training function"""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.student_model
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        student_name = cli_config.student_model.replace("/", "-")
        teacher_name = cli_config.teacher_model.replace("/", "-").split("-")[-1]  # e.g., "30B"
        run_name = (
            f"agentic-distill-{student_name}-from-{teacher_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"kl{cli_config.kl_penalty_coef}-max{cli_config.max_turns}turns-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/rtl-agentic/{run_name}")

    # Create wandb name if not specified
    wandb_name = cli_config.wandb_name or os.path.basename(log_path)

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Load agentic system message
    agentic_msg_path = os.path.join(os.path.dirname(__file__), "AGENTIC_SYSTEM_MESSAGE.txt")
    if os.path.exists(agentic_msg_path):
        with open(agentic_msg_path, 'r') as f:
            override_system_message = f.read().strip()
        logger.info(f"Using agentic system message from {agentic_msg_path}")
    else:
        raise FileNotFoundError(f"Agentic message file not found: {agentic_msg_path}")

    # Create CVDP agentic dataset builder
    dataset_builder = CVDPAgenticDatasetBuilder(
        cvdp_jsonl_path=cli_config.cvdp_jsonl_path,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.student_model,
        renderer_name=renderer_name,
        workspace_dir=cli_config.workspace_dir,
        docker_image=cli_config.docker_image,
        timeout_seconds=cli_config.timeout_seconds,
        max_turns=cli_config.max_turns,
        format_coef=cli_config.format_coef,
        syntax_coef=cli_config.syntax_coef,
        test_coef=cli_config.test_coef,
        test_jsonl_path=cli_config.test_jsonl_path,
        test_split=cli_config.test_split,
        override_system_message=override_system_message,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.batch_size,
    )

    # Create training config
    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.student_model,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,  # type: ignore
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    logger.info("=" * 80)
    logger.info("RTL Code Generation - Agentic On-Policy Distillation")
    logger.info("=" * 80)
    logger.info(f"Student: {cli_config.student_model}")
    logger.info(f"Teacher: {cli_config.teacher_model}")
    logger.info(f"Dataset: {cli_config.cvdp_jsonl_path}")
    logger.info(f"Mode: Multi-turn agentic (max {cli_config.max_turns} turns)")
    logger.info(f"Batch size: {cli_config.batch_size}, Group size: {cli_config.group_size}")
    logger.info(f"Learning rate: {cli_config.learning_rate}, LoRA rank: {cli_config.lora_rank}")
    logger.info(f"Max tokens: {cli_config.max_tokens}")
    logger.info(f"KL penalty coefficient: {cli_config.kl_penalty_coef}")
    logger.info(f"Docker image: {cli_config.docker_image}")
    logger.info(f"Log path: {log_path}")
    logger.info("=" * 80)

    # Run training
    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))
