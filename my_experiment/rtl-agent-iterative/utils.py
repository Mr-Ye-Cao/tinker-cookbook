"""
Utilities for agentic CVDP environment.

Includes:
- Observation formatting
- Output parsing helpers
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def format_command_observation(command: str, stdout: str, stderr: str, returncode: int) -> str:
    """
    Format command execution result as observation for the model.

    Args:
        command: The command that was executed
        stdout: Standard output from command
        stderr: Standard error from command
        returncode: Exit code

    Returns:
        Formatted observation string
    """
    parts = []

    # Show the command that was executed
    parts.append(f"$ {command}")
    parts.append("")

    # Show stdout if available
    if stdout and stdout.strip():
        parts.append(stdout.rstrip())

    # Show stderr if available
    if stderr and stderr.strip():
        if parts[-1]:  # Add separator if stdout was shown
            parts.append("")
        parts.append(stderr.rstrip())

    # Show exit code if non-zero
    if returncode != 0:
        if parts[-1]:
            parts.append("")
        parts.append(f"[Exit code: {returncode}]")

    return "\n".join(parts)


def truncate_output(text: str, max_chars: int = 8000) -> str:
    """
    Truncate long output to prevent context overflow.

    Args:
        text: Text to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated text with message if truncated
    """
    if len(text) <= max_chars:
        return text

    # Keep first 75% and last 25%
    keep_start = int(max_chars * 0.75)
    keep_end = int(max_chars * 0.25)

    truncated = (
        text[:keep_start] +
        f"\n\n[... Output truncated ({len(text) - max_chars} chars omitted) ...]\n\n" +
        text[-keep_end:]
    )

    return truncated
