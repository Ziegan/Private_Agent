"""Permission policy helpers for agent tool calls."""

from collections.abc import Callable
from typing import Any

MUTATING_TOOL_NAMES = frozenset({
    "edit_local_file",
    "apply_workspace_patch",
    "rename_workspace_file",
    "delete_workspace_file",
    "run_shell_command",
    "download_web_file",
    "delete_chat_history_from_sqlite",
    "delete_chat_history_entry_from_sqlite",
    "create_skill",
    "run_project_unit_tests",
    "checkpoint_code_task",
    "finalize_code_task",
})
MONITORED_COMMAND_TOOL_NAMES = frozenset({
    "run_shell_command",
    "run_project_unit_tests",
    "checkpoint_code_task",
    "finalize_code_task",
})
CODING_ACCESS_LEVELS = {
    "full_hitl": (
        "Full workspace access — network-enabled commands; ask before every tool action"
    ),
    "full_governed": (
        "Full workspace access — network-enabled commands; ask before edits and commands"
    ),
    "full_monitored": (
        "Full workspace access — network-enabled commands; approve and monitor each command"
    ),
    "isolated_governed": (
        "Isolated workspace copy — read freely; ask before edits and commands"
    ),
    "isolated_hitl": (
        "Fully isolated workspace copy — ask before every tool action"
    ),
}


def tool_needs_permission(
    name: str,
    permission_mode: str,
    *,
    network_authorized: bool,
    network_tool_names: frozenset[str],
    mcp_tool_names: set[str],
    mcp_auto_approve_tools: Any,
) -> bool:
    if permission_mode == "monitored":
        return (
            name in MONITORED_COMMAND_TOOL_NAMES
            or name in mcp_tool_names and name not in mcp_auto_approve_tools
        )
    if permission_mode == "full":
        return False
    if permission_mode == "manual":
        return name not in network_tool_names or not network_authorized
    return (
        name in MUTATING_TOOL_NAMES
        or name in mcp_tool_names and name not in mcp_auto_approve_tools
    )


def select_coding_access_level(
    console: Any,
    is_interactive: Callable[[], bool],
) -> str:
    """Ask how a coding plan may access its selected workspace."""
    if not is_interactive():
        return "isolated_governed"
    console.print("[bold cyan][Coding Workspace Access][/bold cyan]")
    choices = tuple(CODING_ACCESS_LEVELS.items())
    for index, (_, description) in enumerate(choices, 1):
        console.print(f"  {index}. {description}")
    selection = console.input(
        "Choose workspace access [1-5] (default: 4, isolated): "
    ).strip()
    try:
        choice = int(selection) if selection else 4
    except ValueError:
        console.print("[yellow]Invalid choice; using isolated workspace.[/yellow]")
        choice = 4
    if not 1 <= choice <= len(choices):
        console.print("[yellow]Invalid choice; using isolated workspace.[/yellow]")
        choice = 4
    return choices[choice - 1][0]


def coding_permission_mode(access_level: str) -> str:
    """Translate a reviewed workspace level into the per-task tool policy."""
    return {
        "full_hitl": "manual",
        "full_governed": "auto",
        "full_monitored": "monitored",
        "isolated_governed": "auto",
        "isolated_hitl": "manual",
    }.get(access_level, "auto")


def coding_uses_isolated_copy(access_level: str) -> bool:
    return access_level not in {"full_hitl", "full_governed", "full_monitored"}


def coding_allows_command_network(access_level: str) -> bool:
    return access_level in {"full_hitl", "full_governed", "full_monitored"}


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
    session_consent_state: dict[str, bool] | None = None,
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
    session_consent_state = session_consent_state if session_consent_state is not None else {}
    needs_session_consent = (
        consent_policy == "session"
        and not session_consent_state.get("granted", False)
    )
    needs_per_request_consent = consent_policy == "ask"
    if needs_session_consent or needs_per_request_consent:
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
        if needs_session_consent:
            session_consent_state["granted"] = True
    elif permission_mode == "manual":
        if not is_interactive():
            return False, "Online search requires interactive user approval."
        request_summary = (
            tool_args.get("query")
            or tool_args.get("url")
            or "request details unavailable"
        )
        choice = console.input(
            f"[yellow]Approve this '{tool_name}' action for the current request: "
            f"{request_summary}\nProceed? \\[y/N]: [/yellow]"
        ).strip().lower()
        if choice != "y":
            return False, "Online research was not approved; no request was made."
    return True, ""
