"""
Standard RL Training Script for RTL Code Generation

Trains a model on CVDP benchmark using reinforcement learning.
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

from cvdp_dataset import CVDPDatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for RL training"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset configuration
    cvdp_jsonl_path: str = "/path/to/cvdp_v1.0.1_agentic_code_generation_no_commercial.jsonl"
    test_jsonl_path: str | None = None
    test_split: float = 0.0

    # CVDP configuration
    workspace_dir: str = "/tmp/cvdp_workspace"
    oss_sim_image: str = "ghcr.io/hdl/sim/osv:latest"
    timeout_seconds: int = 300

    # Reward shaping
    format_coef: float = 0.1
    syntax_coef: float = 0.3
    test_coef: float = 1.0

    # Training hyperparameters
    batch_size: int = 8
    group_size: int = 4
    learning_rate: float = 1e-4
    max_tokens: int = 8192

    # Advanced training options
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"
    kl_penalty_coef: float = 0.0  # No KL penalty for standard RL
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
        cli_config.model_name
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"rl-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.batch_size}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/rtl-rl/{run_name}")

    # Create wandb name if not specified
    wandb_name = cli_config.wandb_name or os.path.basename(log_path)

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Create CVDP dataset builder
    dataset_builder = CVDPDatasetBuilder(
        cvdp_jsonl_path=cli_config.cvdp_jsonl_path,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        workspace_dir=cli_config.workspace_dir,
        oss_sim_image=cli_config.oss_sim_image,
        timeout_seconds=cli_config.timeout_seconds,
        format_coef=cli_config.format_coef,
        syntax_coef=cli_config.syntax_coef,
        test_coef=cli_config.test_coef,
        test_jsonl_path=cli_config.test_jsonl_path,
        test_split=cli_config.test_split,
    )

    # Create training config
    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
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
    logger.info("RTL Code Generation - RL Training")
    logger.info("=" * 80)
    logger.info(f"Model: {cli_config.model_name}")
    logger.info(f"Dataset: {cli_config.cvdp_jsonl_path}")
    logger.info(f"Batch size: {cli_config.batch_size}, Group size: {cli_config.group_size}")
    logger.info(f"Learning rate: {cli_config.learning_rate}, LoRA rank: {cli_config.lora_rank}")
    logger.info(f"Log path: {log_path}")
    logger.info("=" * 80)

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))
