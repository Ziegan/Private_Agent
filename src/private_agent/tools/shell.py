"""Approved shell execution inside the configured OS sandbox."""

import shlex
import subprocess
import sys

from langchain_core.tools import tool

from ..config import SHELL_COMMAND_TIMEOUT
from ..sandbox import SandboxManager, evaluate_shell_command
from . import _tool_invocation_approved, console
from .schemas import RunShellInput


@tool(args_schema=RunShellInput)
def run_shell_command(command: str) -> str:
    """Run an approved command inside the configured OS-isolated workspace."""
    try:
        command_args = shlex.split(command)
        if not command_args:
            return "Error: Command must not be empty."
        if not sys.stdin.isatty() and not _tool_invocation_approved.get():
            return (
                "Error: Shell execution requires an interactive terminal and explicit "
                "per-command approval; no command was run."
            )
        risk_notice = (
            " This command matches a high-risk pattern."
            if evaluate_shell_command(command)
            else ""
        )
        from .. import code_tasks

        console.print(
            "[yellow]Commands run in a Linux OS sandbox with "
            + (
                "network access enabled by the approved coding access level, "
                if code_tasks.ACTIVE_CODE_TASK is not None
                and getattr(code_tasks.ACTIVE_CODE_TASK, "allow_network", False)
                else "network disabled, "
            )
            + f"workspace-only writes, and resource limits.{risk_notice}[/yellow]"
        )
        if not _tool_invocation_approved.get():
            approval = console.input(
                f"[yellow]Approve command "
                f"{' '.join(shlex.quote(arg) for arg in command_args)}? "
                "\\[y/N]: [/yellow]"
            ).strip().lower()
            if approval != "y":
                return "Error: Command execution was not approved; no command was run."
        if code_tasks.ACTIVE_CODE_TASK is not None and command_args[0] == "git":
            return (
                "Error: Use checkpoint_code_task for local code-task checkpoints; "
                "manual Git commands are disabled."
            )
        isolated_command, environment = code_tasks._isolated_command(
            command_args,
            code_tasks.ACTIVE_CODE_TASK.root
            if code_tasks.ACTIVE_CODE_TASK is not None
            else SandboxManager.root_dir,
            allow_network=(
                code_tasks.ACTIVE_CODE_TASK is not None
                and getattr(code_tasks.ACTIVE_CODE_TASK, "allow_network", False)
            ),
        )
        result = subprocess.run(
            isolated_command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=SHELL_COMMAND_TIMEOUT,
            env=environment,
        )
        return f"Exit Code: {result.returncode}\nOutput:\n{result.stdout}{result.stderr}"
    except RuntimeError as exc:
        return f"Execution blocked: {exc}"
    except Exception as exc:
        return f"Execution failed: {exc}"
