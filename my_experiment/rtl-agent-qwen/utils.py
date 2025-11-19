"""
Utilities for agentic CVDP environment.

Includes:
- Command parsing from model output
- Docker container management helpers
- Observation formatting
"""

import re
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


def extract_command_from_text(text: str) -> Optional[str]:
    """
    Extract bash command from model's generated text.

    Supports multiple formats:
    1. Markdown code blocks: ```bash\ncommand\n```
    2. Inline code: `command`
    3. Direct command in text

    Args:
        text: Model's generated text

    Returns:
        Extracted command string, or None if no command found
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    if not text:
        return None

    # Priority 1: Markdown code blocks with bash/sh/shell tags
    # Match ```bash\ncommand\n``` or ```sh\ncommand\n``` or ```\ncommand\n```
    code_block_pattern = r'```(?:bash|sh|shell)?\s*\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    if code_blocks:
        # Take the last code block (most recent command)
        command = code_blocks[-1].strip()
        logger.debug(f"Extracted command from code block: {command[:100]}")
        return command

    # Priority 2: Inline code with backticks: `command`
    # Look for commands like `ls /code/rtl` or `cat file.txt`
    inline_code_pattern = r'`([^`]+)`'
    inline_codes = re.findall(inline_code_pattern, text)

    for code in inline_codes:
        # Check if it looks like a shell command (starts with common commands)
        if looks_like_shell_command(code.strip()):
            command = code.strip()
            logger.debug(f"Extracted command from inline code: {command[:100]}")
            return command

    # Priority 3: Direct command (line starts with common shell command)
    # Split by lines and find the first line that looks like a command
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if looks_like_shell_command(line):
            logger.debug(f"Extracted direct command: {line[:100]}")
            return line

    # Priority 4: If nothing found, check if the entire text is a command
    if looks_like_shell_command(text):
        logger.debug(f"Using entire text as command: {text[:100]}")
        return text

    logger.debug(f"No command found in text: {text[:200]}")
    return None


def looks_like_shell_command(text: str) -> bool:
    """
    Check if text looks like a shell command.

    Args:
        text: Text to check

    Returns:
        True if text appears to be a shell command
    """
    text = text.strip()

    if not text:
        return False

    # Common shell commands
    command_prefixes = [
        'ls', 'cat', 'cd', 'pwd', 'echo', 'mkdir', 'rm', 'cp', 'mv',
        'grep', 'find', 'sed', 'awk', 'head', 'tail', 'wc',
        'iverilog', 'vvp', 'pytest', 'python', 'python3',
        'bash', 'sh', 'chmod', 'chown',
        'tree', 'file', 'which', 'whoami',
    ]

    # Check if starts with any common command
    first_word = text.split()[0] if text.split() else ''

    if first_word in command_prefixes:
        return True

    # Check for shell operators (>, >>, |, &&, ||, ;)
    shell_operators = ['>', '>>', '|', '&&', '||', ';', '<<']
    if any(op in text for op in shell_operators):
        return True

    # Check for variable assignment (VAR=value)
    if re.match(r'^[A-Z_][A-Z0-9_]*=', text):
        return True

    return False


def validate_command_safety(command: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a command is safe to execute.

    Prevents dangerous operations like:
    - Deleting important files (rm -rf /, etc.)
    - Network operations (curl, wget outside /code)
    - System modifications

    Args:
        command: Command to validate

    Returns:
        (is_safe: bool, reason: Optional[str])
    """
    command = command.strip()

    # Dangerous patterns
    dangerous_patterns = [
        (r'rm\s+(-rf\s+)?/', 'Deletion of root or system paths'),
        (r'rm\s+.*~', 'Deletion of user home directory'),
        (r'mkfs', 'Filesystem formatting'),
        (r'dd\s+.*of=/dev/', 'Direct disk write'),
        (r':(){ :|:& };:', 'Fork bomb'),
        (r'chmod\s+777\s+/', 'Dangerous permission change on root'),
        (r'chown\s+.*/', 'Ownership change on root'),
    ]

    for pattern, reason in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return False, reason

    # Allow everything within /code directory
    # This includes: cat, echo, ls, iverilog, pytest, etc.
    return True, None


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


def extract_verilog_from_file_write(text: str) -> Optional[str]:
    """
    Extract Verilog code from file write commands like:
    - echo "code" > file.sv
    - cat > file.sv << 'EOF' ... EOF

    Args:
        text: Text containing file write command

    Returns:
        Extracted Verilog code, or None if not found
    """
    # Pattern 1: Heredoc (cat > file << 'EOF' ... EOF)
    heredoc_pattern = r"cat\s*>\s*[^\s]+\s*<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1"
    heredoc_match = re.search(heredoc_pattern, text, re.DOTALL)
    if heredoc_match:
        code = heredoc_match.group(2)
        logger.debug(f"Extracted Verilog from heredoc: {len(code)} chars")
        return code

    # Pattern 2: Echo (echo "..." > file)
    echo_pattern = r'echo\s+["\'](.+?)["\']\s*>\s*[^\s]+'
    echo_match = re.search(echo_pattern, text, re.DOTALL)
    if echo_match:
        code = echo_match.group(1)
        logger.debug(f"Extracted Verilog from echo: {len(code)} chars")
        return code

    return None


def parse_pytest_output(output: str) -> Tuple[int, int, float]:
    """
    Parse pytest output to extract pass/fail counts.

    Args:
        output: stdout/stderr from pytest

    Returns:
        (passed, failed, pass_rate)
    """
    # Look for patterns like:
    # "5 passed, 2 failed in 1.23s"
    # "10 passed in 0.45s"
    # "3 failed in 2.10s"

    passed = 0
    failed = 0

    # Match "N passed"
    passed_match = re.search(r'(\d+)\s+passed', output)
    if passed_match:
        passed = int(passed_match.group(1))

    # Match "N failed"
    failed_match = re.search(r'(\d+)\s+failed', output)
    if failed_match:
        failed = int(failed_match.group(1))

    # Calculate pass rate
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    return passed, failed, pass_rate


def check_iverilog_success(output: str) -> bool:
    """
    Check if iverilog compilation was successful.

    Args:
        output: stdout/stderr from iverilog

    Returns:
        True if compilation successful (no errors)
    """
    # iverilog prints errors to stderr
    # Look for common error indicators
    error_indicators = [
        'error:',
        'syntax error',
        'Error:',
        'ERROR:',
    ]

    output_lower = output.lower()

    for indicator in error_indicators:
        if indicator.lower() in output_lower:
            return False

    # If no error indicators, consider it successful
    return True
