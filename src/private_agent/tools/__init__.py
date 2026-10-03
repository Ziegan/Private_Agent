import os
import socket
import contextvars
from contextlib import contextmanager
from typing import Optional, Callable
from functools import wraps
from datetime import datetime
from langchain_core.tools import tool
from rich.console import Console

from .schemas import (
    CreateSkillInput,
)
from . import _sqlite_state
from .network import (
    _PublicOnlyAsyncHTTPTransport as _PublicOnlyAsyncHTTPTransport,
    _PublicOnlyNetworkBackend as _PublicOnlyNetworkBackend,
    _PublicOnlySyncHTTPTransport as _PublicOnlySyncHTTPTransport,
    _PublicOnlySyncNetworkBackend as _PublicOnlySyncNetworkBackend,
    _get_limited_response as _get_limited_response,
    _network_slots as _network_slots,
    _outbound_http_clients as _outbound_http_clients,
    _public_only_async_client as _public_only_async_client,
    _resolve_validated_peer as _resolve_validated_peer,
    close_outbound_http_clients as close_outbound_http_clients,
    ollama_client_kwargs as ollama_client_kwargs,
    ollama_langchain_client_kwargs as ollama_langchain_client_kwargs,
    public_only_sync_client as public_only_sync_client,
    track_ollama_http_clients as track_ollama_http_clients,
    search_duckduckgo,
    validate_explicit_service_url,
    validate_outbound_url as validate_outbound_url,
)
from .media import (
    capture_webcam_image,
    list_microphone_devices,
    record_microphone_audio,
)
from ..config import (
    INTERNET_CHECK_HOST,
    INTERNET_CHECK_PORT,
    INTERNET_CHECK_TIMEOUT,
    SKILL_DESCRIPTION_PREVIEW_CHARS,
    MCP_AUTO_APPROVE_TOOLS,
    SKILLS_FOLDER_DEFAULT,
)

console = Console()
_search_duckduckgo = search_duckduckgo
_validate_explicit_service_url = validate_explicit_service_url

_sqlite_lock = _sqlite_state.SQLITE_LOCK
_tool_invocation_approved = contextvars.ContextVar(
    "private_agent_tool_invocation_approved", default=False
)


@contextmanager
def approved_tool_invocation():
    """Mark the current model tool call as approved by the session policy."""
    token = _tool_invocation_approved.set(True)
    try:
        yield
    finally:
        _tool_invocation_approved.reset(token)

def set_active_db_path(path: str):
    _sqlite_state.set_active_db_path(path)

# --- NETWORK CONNECTIVITY UTILITY ---
def check_internet_connection(
    host: str = INTERNET_CHECK_HOST,
    port: int = INTERNET_CHECK_PORT,
    timeout: float = INTERNET_CHECK_TIMEOUT,
) -> bool:
    """Quick socket probe to check active TCP-level internet reachability."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True
    except OSError:
        return False


NETWORK_TOOL_NAMES = frozenset({"web_search", "fetch_webpage", "download_web_file"})
LOCAL_MEDIA_TOOL_NAMES = frozenset({
    "capture_webcam_image",
    "list_microphone_devices",
    "record_microphone_audio",
})
MCP_TOOL_NAMES = set()
MCP_TOOL_SOURCES = {}
MCP_TOOL_TRANSPORTS = {}
_ACTIVE_SKILL_DIRECTORY = os.path.abspath(
    os.path.expanduser(SKILLS_FOLDER_DEFAULT)
) if SKILLS_FOLDER_DEFAULT else None
_ACTIVE_SKILL_REGISTRY = None


def set_active_skill_runtime(folder_path: Optional[str], loaded_skills: dict) -> None:
    """Set the configured destination and in-memory registry for skill creation."""
    global _ACTIVE_SKILL_DIRECTORY, _ACTIVE_SKILL_REGISTRY
    _ACTIVE_SKILL_DIRECTORY = (
        os.path.abspath(os.path.expanduser(folder_path)) if folder_path else None
    )
    _ACTIVE_SKILL_REGISTRY = loaded_skills


def describe_tool_catalog() -> list[dict[str, str]]:
    """Return a user-facing inventory of each tool's origin and approval boundary."""
    catalog = []
    auto_approved_names = (
        {str(name) for name in MCP_AUTO_APPROVE_TOOLS}
        if isinstance(MCP_AUTO_APPROVE_TOOLS, (set, list, tuple))
        else set()
    )
    for name, tool_object in sorted(AVAILABLE_TOOLS.items()):
        is_mcp = name in MCP_TOOL_NAMES
        origin = MCP_TOOL_SOURCES.get(name, "configured MCP server") if is_mcp else "built-in"
        if is_mcp:
            permission = (
                "auto-approved by configuration"
                if name in auto_approved_names
                else "approval required"
            )
        elif name in NETWORK_TOOL_NAMES:
            permission = "network consent required"
        elif name == "create_skill":
            permission = "interactive approval required before creating a skill"
        elif name in LOCAL_MEDIA_TOOL_NAMES:
            permission = (
                "local device metadata only"
                if name == "list_microphone_devices"
                else "explicit per-use local capture approval"
            )
        elif name == "get_local_datetime":
            permission = "no approval required; reads local system clock"
        elif name in {"run_shell_command", "delete_chat_history_from_sqlite"}:
            permission = "interactive approval for shell/deletion/network actions"
        else:
            permission = "workspace path validation"

        if name in NETWORK_TOOL_NAMES:
            effects = "network"
        elif name in LOCAL_MEDIA_TOOL_NAMES:
            effects = (
                "local microphone device metadata"
                if name == "list_microphone_devices"
                else "may access a local media device"
            )
        elif name in {
            "edit_local_file",
            "run_shell_command",
            "download_web_file",
            "delete_chat_history_from_sqlite",
            "create_skill",
        }:
            effects = "may modify local state"
        elif is_mcp:
            effects = (
                "network-capable; server-defined effects"
                if MCP_TOOL_TRANSPORTS.get(name) != "stdio"
                else "server-defined; inspect trusted MCP server"
            )
        else:
            effects = "read-only/local"
        catalog.append({
            "name": name,
            "origin": origin,
            "permission": permission,
            "effects": effects,
            "description": str(getattr(tool_object, "description", "") or "").splitlines()[0][:120],
        })
    return catalog


def has_internet_connection() -> bool:
    return check_internet_connection(
        INTERNET_CHECK_HOST, INTERNET_CHECK_PORT, INTERNET_CHECK_TIMEOUT
    )

def require_internet(func: Callable) -> Callable:
    """Decorator that validates active internet connection before executing a web-bound tool."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_internet_connection():
            return "Error: Network unavailable. Please check your internet connection and try again."
        return func(*args, **kwargs)
    return wrapper


# --- TOOL IMPLEMENTATIONS ---
@tool
def get_local_datetime() -> str:
    """Get the current local date, time, weekday, month, and year from this machine."""
    now = datetime.now().astimezone()
    return (
        f"Local date: {now:%A, }{now.day} {now:%B %Y}\n"
        f"Local time: {now:%H:%M:%S %Z} (UTC{now:%z})\n"
        f"Weekday: {now:%A}\n"
        f"Month: {now:%B}\n"
        f"Year: {now:%Y}"
    )


@tool(args_schema=CreateSkillInput)
def create_skill(name: str, description: str, instructions: str) -> str:
    """Create a reusable Markdown skill in the configured skills folder after user approval."""
    from ..skills import AgentSkill, create_skill_file

    try:
        if _ACTIVE_SKILL_DIRECTORY is None:
            return "Error creating skill: No skills directory is configured."
        target = create_skill_file(
            _ACTIVE_SKILL_DIRECTORY, name, description, instructions
        )
        skill_key = target.stem
        display_name = name.strip()
        skill = AgentSkill(
            name=display_name,
            description=(
                f"{description.strip()} "
                f"{instructions.strip()[:SKILL_DESCRIPTION_PREVIEW_CHARS]}"
            ),
            system_prompt=f"[Skill Markdown Profile: {display_name}]\n"
            f"{target.read_text(encoding='utf-8')}",
        )
        if isinstance(_ACTIVE_SKILL_REGISTRY, dict):
            _ACTIVE_SKILL_REGISTRY[skill_key] = skill
        return (
            f"Created skill '{display_name}' at {target}. "
            "It is available in this session and future runs."
        )
    except (OSError, ValueError) as exc:
        return f"Error creating skill: {exc}"


# Import after network helpers are defined; web.py depends on this shared policy layer.
from .web import download_web_file, fetch_webpage, web_search  # noqa: E402
from .filesystem import edit_local_file, list_directory, read_local_file  # noqa: E402
from .shell import run_shell_command  # noqa: E402
from .memory import (  # noqa: E402
    delete_chat_history_from_sqlite,
    read_chat_history_from_sqlite,
)


AVAILABLE_TOOLS = {
    "get_local_datetime": get_local_datetime,
    "read_local_file": read_local_file,
    "edit_local_file": edit_local_file,
    "run_shell_command": run_shell_command,
    "web_search": web_search,
    "list_directory": list_directory,
    "fetch_webpage": fetch_webpage,
    "download_web_file": download_web_file,
    "read_chat_history_from_sqlite": read_chat_history_from_sqlite,
    "delete_chat_history_from_sqlite": delete_chat_history_from_sqlite,
    "create_skill": create_skill,
    "capture_webcam_image": capture_webcam_image,
    "list_microphone_devices": list_microphone_devices,
    "record_microphone_audio": record_microphone_audio,
}

async def load_mcp_tools(*args, **kwargs) -> list:
    """
    Connects to an MCP server, retrieves available tools, and returns them as a list.
    Guarantees that an empty list ([]) is returned upon any connection error, 
    timeout, or exception instead of a dictionary or None.
    """
    try:
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession
        
        # Establishing connection parameters and session
        async with stdio_client(*args, **kwargs) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                
                tools = []
                for tool in getattr(response, "tools", []):
                    tools.append(tool)
                return tools
    except Exception as exc:
        console.print(f"[red][MCP error] Unable to discover tools: {exc}[/red]")
        return []


from .mcp import (  # noqa: E402
    close_mcp_sandbox_dirs as close_mcp_sandbox_dirs,
    load_configured_mcp_tools as load_configured_mcp_tools,
)
