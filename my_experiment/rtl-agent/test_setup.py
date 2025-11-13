"""
Test script to validate CVDP + Tinker integration.

This script tests the environment without actually running training.
"""

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cvdp_env import CVDPEnv
from cvdp_dataset import CVDPDatasetBuilder
from utils import load_cvdp_problems, get_cvdp_statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_environment_creation():
    """Test creating a CVDP environment"""
    logger.info("Testing environment creation...")

    # Create a simple test problem
    test_problem = {
        "id": "test_problem_001",
        "prompt": "Design a simple AND gate module in Verilog.",
        "context": {
            "docs/spec.md": "# AND Gate\nCreate a 2-input AND gate.",
        },
        "harness": {
            "docker-compose.yml": """services:
  direct:
    image: ghcr.io/hdl/sim/osv:latest
    command: echo "Mock test"
""",
        },
    }

    # Get tokenizer and renderer
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")
    renderer = renderers.get_renderer("role_colon", tokenizer)

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create environment
        env = CVDPEnv(
            problem_id=test_problem["id"],
            prompt=test_problem["prompt"],
            context_files=test_problem["context"],
            harness_config=test_problem["harness"],
            workspace_dir=tmpdir,
            renderer=renderer,
        )

        # Test initial observation
        obs, stop = await env.initial_observation()
        logger.info(f"✓ Initial observation created: {len(obs.tokens)} tokens")

        # Check workspace was created
        workspace_path = Path(tmpdir) / test_problem["id"]
        assert workspace_path.exists(), "Workspace not created"
        logger.info(f"✓ Workspace created at: {workspace_path}")

        # Check files were written
        assert (workspace_path / "prompt.json").exists(), "prompt.json not created"
        assert (workspace_path / "docs/spec.md").exists(), "Context files not written"
        logger.info("✓ All expected files created")

    logger.info("✅ Environment creation test passed!\n")


async def test_dataset_loading():
    """Test loading CVDP dataset"""
    logger.info("Testing dataset loading...")

    # Check if example dataset exists
    cvdp_benchmark_path = Path("/Users/yecao/Downloads/project/cvdp_benchmark")
    example_dataset = cvdp_benchmark_path / "example_dataset" / "cvdp_v1.0.1_example_agentic_code_generation_no_commercial_with_solutions.jsonl"

    if not example_dataset.exists():
        logger.warning(f"⚠️  Example dataset not found at {example_dataset}")
        logger.warning("Skipping dataset loading test")
        return

    # Load problems
    problems = load_cvdp_problems(str(example_dataset), max_problems=5)
    logger.info(f"✓ Loaded {len(problems)} problems")

    # Get statistics
    stats = get_cvdp_statistics(problems)
    logger.info(f"✓ Dataset statistics:")
    logger.info(f"  - Total problems: {stats['total_problems']}")
    logger.info(f"  - Categories: {stats['categories']}")
    logger.info(f"  - Avg prompt length: {stats['avg_prompt_length']:.0f} chars")

    # Test dataset builder (without actually building)
    builder = CVDPDatasetBuilder(
        cvdp_jsonl_path=str(example_dataset),
        batch_size=2,
        group_size=2,
        model_name_for_tokenizer="meta-llama/Llama-3.2-1B",
        renderer_name="role_colon",
    )
    logger.info("✓ Dataset builder created")

    logger.info("✅ Dataset loading test passed!\n")


async def test_code_extraction():
    """Test Verilog code extraction"""
    logger.info("Testing code extraction...")

    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")
    renderer = renderers.get_renderer("role_colon", tokenizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        env = CVDPEnv(
            problem_id="test",
            prompt="Test",
            context_files={},
            harness_config={},
            workspace_dir=tmpdir,
            renderer=renderer,
        )

        # Test different code block formats
        test_cases = [
            (
                "Here's the code:\n```verilog\nmodule test;\nendmodule\n```",
                "module test;\nendmodule",
                "verilog"
            ),
            (
                "```systemverilog\nmodule test2;\nendmodule\n```",
                "module test2;\nendmodule",
                "systemverilog"
            ),
            (
                "```sv\nmodule test3;\nendmodule\n```",
                "module test3;\nendmodule",
                "sv"
            ),
            (
                "No code block here",
                None,
                "no_block"
            ),
        ]

        for text, expected, label in test_cases:
            result = env._extract_verilog(text)
            if expected is None:
                assert result is None, f"{label}: Should return None"
                logger.info(f"✓ {label}: Correctly returned None")
            else:
                assert result == expected, f"{label}: Mismatch"
                logger.info(f"✓ {label}: Extracted correctly")

    logger.info("✅ Code extraction test passed!\n")


async def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("CVDP + Tinker Integration Test Suite")
    logger.info("=" * 80)
    logger.info("")

    try:
        await test_environment_creation()
        await test_dataset_loading()
        await test_code_extraction()

        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Prepare a small CVDP dataset (5-10 problems)")
        logger.info("2. Run: python train_rl.py --cvdp_jsonl_path /path/to/dataset.jsonl")
        logger.info("3. Monitor training logs and metrics")

    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
