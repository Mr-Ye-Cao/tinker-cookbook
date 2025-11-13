"""
Utility functions for CVDP + Tinker integration.
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_cvdp_problems(jsonl_path: str, max_problems: int | None = None) -> List[Dict]:
    """
    Load CVDP problems from JSONL file.

    Args:
        jsonl_path: Path to CVDP JSONL file
        max_problems: Maximum number of problems to load (None = all)

    Returns:
        List of problem dictionaries
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"CVDP JSONL not found: {jsonl_path}")

    problems = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if max_problems and len(problems) >= max_problems:
                break

            line = line.strip()
            if not line:
                continue

            try:
                problem = json.loads(line)
                problems.append(problem)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
                continue

    logger.info(f"Loaded {len(problems)} problems from {jsonl_path}")
    return problems


def filter_cvdp_problems_by_category(
    problems: List[Dict],
    categories: List[str] | None = None,
    exclude_categories: List[str] | None = None,
) -> List[Dict]:
    """
    Filter CVDP problems by category.

    Args:
        problems: List of problem dictionaries
        categories: List of categories to include (None = all)
        exclude_categories: List of categories to exclude

    Returns:
        Filtered list of problems
    """
    filtered = []

    for problem in problems:
        problem_categories = problem.get("categories", [])

        # Check inclusion
        if categories:
            if not any(cat in problem_categories for cat in categories):
                continue

        # Check exclusion
        if exclude_categories:
            if any(cat in problem_categories for cat in exclude_categories):
                continue

        filtered.append(problem)

    logger.info(f"Filtered {len(filtered)}/{len(problems)} problems by category")
    return filtered


def get_cvdp_statistics(problems: List[Dict]) -> Dict:
    """
    Get statistics about CVDP problems.

    Args:
        problems: List of problem dictionaries

    Returns:
        Dictionary with statistics
    """
    categories = {}
    prompt_lengths = []
    has_harness = 0
    has_context = 0

    for problem in problems:
        # Count categories
        for cat in problem.get("categories", []):
            categories[cat] = categories.get(cat, 0) + 1

        # Prompt lengths
        prompt_lengths.append(len(problem.get("prompt", "")))

        # Harness/context availability
        if problem.get("harness"):
            has_harness += 1
        if problem.get("context"):
            has_context += 1

    return {
        "total_problems": len(problems),
        "categories": categories,
        "avg_prompt_length": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "max_prompt_length": max(prompt_lengths) if prompt_lengths else 0,
        "problems_with_harness": has_harness,
        "problems_with_context": has_context,
    }


def create_small_cvdp_subset(
    input_jsonl: str,
    output_jsonl: str,
    num_problems: int = 10,
    categories: List[str] | None = None,
):
    """
    Create a small subset of CVDP problems for testing.

    Args:
        input_jsonl: Input CVDP JSONL file
        output_jsonl: Output JSONL file
        num_problems: Number of problems to include
        categories: Optional category filter
    """
    problems = load_cvdp_problems(input_jsonl)

    if categories:
        problems = filter_cvdp_problems_by_category(problems, categories=categories)

    # Take first num_problems
    subset = problems[:num_problems]

    # Write to output
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w') as f:
        for problem in subset:
            f.write(json.dumps(problem) + '\n')

    logger.info(f"Created subset with {len(subset)} problems: {output_jsonl}")
    return subset


def analyze_cvdp_results(metrics_jsonl: str) -> Dict:
    """
    Analyze training results from metrics.jsonl.

    Args:
        metrics_jsonl: Path to metrics.jsonl from training

    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(metrics_jsonl):
        raise FileNotFoundError(f"Metrics file not found: {metrics_jsonl}")

    format_rates = []
    syntax_rates = []
    pass_rates = []
    rewards = []

    with open(metrics_jsonl, 'r') as f:
        for line in f:
            try:
                metrics = json.loads(line)

                # Extract metrics
                if "format" in metrics:
                    format_rates.append(metrics["format"])
                if "syntax_valid" in metrics:
                    syntax_rates.append(metrics["syntax_valid"])
                if "correct" in metrics or "tests_passed" in metrics:
                    pass_rates.append(metrics.get("correct", metrics.get("tests_passed", 0)))
                if "reward" in metrics:
                    rewards.append(metrics["reward"])

            except json.JSONDecodeError:
                continue

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "total_episodes": len(rewards),
        "avg_format_rate": safe_avg(format_rates),
        "avg_syntax_rate": safe_avg(syntax_rates),
        "avg_pass_rate": safe_avg(pass_rates),
        "avg_reward": safe_avg(rewards),
        "max_reward": max(rewards) if rewards else 0.0,
    }
