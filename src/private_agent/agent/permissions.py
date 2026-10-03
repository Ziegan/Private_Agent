"""Permission policy helpers for agent tool calls."""

from collections.abc import Callable
from typing import Any

MUTATING_TOOL_NAMES = frozenset({
    "edit_local_file",
    "run_shell_command",
    "download_web_file",
    "delete_chat_history_from_sqlite",
    "create_skill",
    "run_project_unit_tests",
    "checkpoint_code_task",
    "finalize_code_task",
})


def tool_needs_permission(
    name: str,
    permission_mode: str,
    *,
    network_authorized: bool,
    network_tool_names: frozenset[str],
    mcp_tool_names: set[str],
    mcp_auto_approve_tools: Any,
) -> bool:
    if permission_mode == "full":
        return False
    if permission_mode == "manual":
        return name not in network_tool_names or not network_authorized
    return (
        name in MUTATING_TOOL_NAMES
        or name in mcp_tool_names and name not in mcp_auto_approve_tools
    )


def select_permission_mode(
    configured: str,
    console: Any,
    is_interactive: Callable[[], bool],
) -> str:
    labels = {
        "manual": "Manual — ask before every tool action",
        "auto": "Auto — read/search freely; ask before edits and command execution",
        "full": "Full — approve all tool calls for this session",
    }
    if not is_interactive():
        return configured
    console.print("[bold cyan][Agent Permission Mode][/bold cyan]")
    for number, mode in enumerate(("manual", "auto", "full"), 1):
        console.print(f"  {number}. {labels[mode]}")
    selection = console.input(
        f"Choose permission mode [1/2/3] (default: {configured}): "
    ).strip().lower()
    if not selection:
        return configured
    modes = {"1": "manual", "2": "auto", "3": "full"}
    return modes.get(selection, configured)


async def authorize_network_research(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    permission_mode: str,
    enabled: bool,
    consent_policy: str,
    internet_check: Callable[[], bool],
    console: Any,
    is_interactive: Callable[[], bool],
) -> tuple[bool, str]:
    """Require live connectivity and consent before an external request."""
    if not enabled:
        return False, "Online research is disabled in configuration."

    if not internet_check():
        if not is_interactive():
            return False, (
                "Internet connectivity is unavailable. Offline evidence may be "
                "incomplete; no web request was made."
            )
        choice = console.input(
            "[yellow]Internet is unavailable. Restore access and retry (r), "
            "continue with incomplete offline data (c), or cancel (x)? [/yellow]"
        ).strip().lower()
        if choice == "r":
            if not internet_check():
                return False, (
                    "Internet connectivity is still unavailable. No web request "
                    "was made; offline evidence may be incomplete."
                )
        elif choice == "c":
            return False, (
                "The user chose to continue without online research. Offline "
                "evidence may be incomplete; do not claim current facts were verified."
            )
        else:
            return False, "Online research was cancelled by the user."

    if consent_policy == "never":
        return False, "Online research is disabled by the configured consent policy."
    if permission_mode == "manual":
        if not is_interactive():
            return False, "Online search requires interactive user consent."
        request_summary = (
            tool_args.get("query")
            or tool_args.get("url")
            or "request details unavailable"
        )
        choice = console.input(
            f"[yellow]'{tool_name}' will send this to an external service: "
            f"{request_summary}\nProceed? \\[y/N]: [/yellow]"
        ).strip().lower()
        if choice != "y":
            return False, "Online research was not approved; use offline evidence only."
    return True, ""
