import os
import sys
import socket
import shlex
import sqlite3
import threading
import pathlib
import contextvars
from contextlib import contextmanager
from typing import Optional, Callable
from functools import wraps
import asyncio
import ipaddress
import tempfile
import urllib.parse
import shutil
import httpx
import httpcore
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from rich.console import Console

from ..sandbox import SandboxManager, create_hitl_snapshot, evaluate_shell_command
from .media_tools import (
    capture_webcam_image,
    list_microphone_devices,
    record_microphone_audio,
)
from ..config import (
    DEFAULT_DB_PATH,
    INTERNET_CHECK_HOST,
    INTERNET_CHECK_PORT,
    INTERNET_CHECK_TIMEOUT,
    MAX_NETWORK_CONCURRENCY,
    NETWORK_REQUEST_TIMEOUT,
    MAX_STREAM_TIMEOUT,
    MAX_WEBPAGE_BYTES,
    MAX_DOWNLOAD_BYTES,
    MAX_HTTP_REDIRECTS,
    MAX_SEARCH_QUERY_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_FETCHED_WEBPAGE_CHARS,
    MAX_READ_FILE_BYTES,
    FILE_READ_CHUNK_BYTES,
    BINARY_PROBE_BYTES,
    SHELL_COMMAND_TIMEOUT,
    DEFAULT_HISTORY_READ_LIMIT,
    SKILL_DESCRIPTION_PREVIEW_CHARS,
    MAX_SKILL_NAME_CHARS,
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTION_CHARS,
    MCP_AUTO_APPROVE_TOOLS,
    SKILLS_FOLDER_DEFAULT,
)

console = Console()

# Global reference for active sqlite db path used by sqlite tools
ACTIVE_DB_PATH = DEFAULT_DB_PATH
_sqlite_lock = threading.Lock()
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
    global ACTIVE_DB_PATH
    ACTIVE_DB_PATH = os.path.abspath(path)

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


_network_slots = threading.BoundedSemaphore(MAX_NETWORK_CONCURRENCY)
_outbound_http_clients = []
_mcp_sandbox_dirs = []
_EXPLICIT_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def _resolve_validated_peer(host: str, port: int, allow_loopback: bool):
    records = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    addresses = []
    explicit_local_host = host.lower() == "localhost"
    literal_address = None
    try:
        literal_address = ipaddress.ip_address(host.split("%", 1)[0])
        explicit_local_host = explicit_local_host or (
            literal_address.is_loopback
            or any(
                literal_address.version == network.version
                and literal_address in network
                for network in _EXPLICIT_PRIVATE_NETWORKS
            )
        )
    except ValueError:
        pass
    for record in records:
        address = ipaddress.ip_address(record[4][0].split("%", 1)[0])
        if not address.is_global and not (
            allow_loopback
            and explicit_local_host
            and (
                address.is_loopback
                if literal_address is None
                else address == literal_address
            )
        ):
            raise OSError(
                f"Outbound connection to non-public address {address} was blocked."
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise OSError("Outbound hostname resolved to no addresses.")
    return addresses


class _PublicOnlyNetworkBackend:
    """Resolve and validate each peer immediately before connecting to its IP."""

    def __init__(self, allow_loopback: bool = False):
        from httpcore._backends.auto import AutoBackend
        self._backend = AutoBackend()
        self._allow_loopback = allow_loopback

    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        addresses = await asyncio.to_thread(
            _resolve_validated_peer, host, port, self._allow_loopback
        )
        last_error = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (OSError, httpcore.ConnectError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("Outbound hostname resolved to no connectable addresses.")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("Unix-domain sockets are disabled for outbound web requests.")

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


class _PublicOnlySyncNetworkBackend:
    def __init__(self, allow_loopback: bool = False):
        from httpcore._backends.sync import SyncBackend
        self._backend = SyncBackend()
        self._allow_loopback = allow_loopback

    def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        addresses = _resolve_validated_peer(host, port, self._allow_loopback)
        last_error = None
        for address in addresses:
            try:
                return self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (OSError, httpcore.ConnectError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("Outbound hostname resolved to no connectable addresses.")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("Unix-domain sockets are disabled for outbound requests.")

    def sleep(self, seconds):
        return self._backend.sleep(seconds)


class _PublicOnlyAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, allow_loopback: bool = False):
        limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
        super().__init__(trust_env=False, limits=limits, retries=0)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PublicOnlyNetworkBackend(allow_loopback),
        )


class _PublicOnlySyncHTTPTransport(httpx.HTTPTransport):
    def __init__(self, allow_loopback: bool = False):
        limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
        super().__init__(trust_env=False, limits=limits, retries=0)
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PublicOnlySyncNetworkBackend(allow_loopback),
        )


def _public_only_async_client(
    *,
    headers: dict,
    timeout: float,
    allow_loopback: bool = False,
    track: bool = False,
):
    client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout),
        transport=_PublicOnlyAsyncHTTPTransport(allow_loopback),
        follow_redirects=False,
        trust_env=False,
    )
    if track:
        _outbound_http_clients.append(client)
    return client


def public_only_sync_client(
    *, headers: dict, timeout: float, allow_loopback=False, track: bool = True
):
    client = httpx.Client(
        headers=headers,
        timeout=timeout,
        transport=_PublicOnlySyncHTTPTransport(allow_loopback),
        follow_redirects=False,
        trust_env=False,
    )
    if track:
        _outbound_http_clients.append(client)
    return client


def ollama_client_kwargs() -> dict:
    """Return Ollama SDK kwargs using peer-validated transports."""
    return {
        "transport": _PublicOnlySyncHTTPTransport(allow_loopback=True),
        "timeout": httpx.Timeout(MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT),
        "follow_redirects": False,
        "trust_env": False,
    }


def ollama_langchain_client_kwargs() -> dict:
    """Return LangChain Ollama SDK custom sync/async HTTP client arguments."""
    return {
        "sync_client_kwargs": {
            "transport": _PublicOnlySyncHTTPTransport(allow_loopback=True),
            "timeout": httpx.Timeout(MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT),
            "follow_redirects": False,
            "trust_env": False,
        },
        "async_client_kwargs": {
            "transport": _PublicOnlyAsyncHTTPTransport(allow_loopback=True),
            "timeout": httpx.Timeout(MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT),
            "follow_redirects": False,
            "trust_env": False,
        },
    }


def track_ollama_http_clients(owner):
    """Track the HTTPX clients created internally by Ollama SDK wrappers."""
    for attribute in ("_client", "_async_client"):
        client = getattr(owner, attribute, None)
        for _ in range(2):
            if isinstance(client, (httpx.Client, httpx.AsyncClient)):
                break
            client = getattr(client, "_client", None)
        if isinstance(client, (httpx.Client, httpx.AsyncClient)):
            if client not in _outbound_http_clients:
                _outbound_http_clients.append(client)


def _validate_explicit_service_url(url: str, transports: set[str]) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in transports
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("Service URL must use an allowed scheme and contain no credentials or fragment.")
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    if parsed.scheme in {"http", "https"}:
        _resolve_validated_peer(parsed.hostname, port, allow_loopback=True)
    else:
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as exc:
            raise ValueError(
                "WebSocket MCP requires an explicit IP address because its "
                "transport cannot pin hostname resolution."
            ) from exc
        if not (
            address.is_global
            or address.is_loopback
            or any(
                address.version == network.version and address in network
                for network in _EXPLICIT_PRIVATE_NETWORKS
            )
        ):
            raise ValueError("WebSocket MCP address is not permitted.")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def _mcp_http_client_factory(headers=None, timeout=None, auth=None):
    limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
    transport = _PublicOnlyAsyncHTTPTransport(allow_loopback=True)
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(NETWORK_REQUEST_TIMEOUT),
        auth=auth,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        limits=limits,
    )


async def close_outbound_http_clients():
    clients, _outbound_http_clients[:] = list(_outbound_http_clients), []
    for client in clients:
        try:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
            else:
                await asyncio.to_thread(client.close)
        except Exception as exc:
            console.print(
                f"[yellow][Shutdown warning] Could not close an outbound "
                f"HTTP client: {type(exc).__name__}[/yellow]"
            )


def _remove_mcp_sandbox_dir(directory: str):
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    except OSError as exc:
        console.print(
            f"[yellow][Shutdown warning] Could not remove MCP workspace "
            f"'{directory}': {type(exc).__name__}[/yellow]"
        )
        return
    if directory in _mcp_sandbox_dirs:
        _mcp_sandbox_dirs.remove(directory)


def close_mcp_sandbox_dirs():
    for directory in list(_mcp_sandbox_dirs):
        _remove_mcp_sandbox_dir(directory)


def validate_outbound_url(url: str) -> str:
    """Reject non-HTTP(S), credentialed, local, and non-public network targets."""
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only HTTP(S) URLs with a host are allowed.")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed.")
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        hostname = parsed.hostname.rstrip(".")
        try:
            addresses = {ipaddress.ip_address(hostname)}
        except ValueError:
            records = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
            addresses = {
                ipaddress.ip_address(record[4][0].split("%", 1)[0])
                for record in records
            }
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("URLs resolving to non-public network addresses are not allowed.")
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unsafe outbound URL: {exc}") from exc
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


async def _get_limited_response(client, url: str, max_bytes: int) -> tuple[bytes, str]:
    acquired = await asyncio.to_thread(
        _network_slots.acquire, True, NETWORK_REQUEST_TIMEOUT
    )
    if not acquired:
        raise TimeoutError("Outbound request concurrency limit wait timed out.")
    try:
        async def _request_with_redirects():
            current_url = await asyncio.to_thread(validate_outbound_url, url)
            for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_HTTP_REDIRECTS:
                            raise ValueError("HTTP redirect limit exceeded or redirect target missing.")
                        current_url = await asyncio.to_thread(
                            validate_outbound_url,
                            urllib.parse.urljoin(current_url, location),
                        )
                        continue
                    response.raise_for_status()
                    declared_size = response.headers.get("content-length")
                    if declared_size and int(declared_size) > max_bytes:
                        raise ValueError(f"Response exceeds the {max_bytes}-byte size limit.")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise ValueError(f"Response exceeds the {max_bytes}-byte size limit.")
                    return bytes(body), response.encoding or "utf-8"
            raise ValueError("HTTP redirect limit exceeded.")
        return await asyncio.wait_for(
            _request_with_redirects(),
            timeout=NETWORK_REQUEST_TIMEOUT,
        )
    finally:
        _network_slots.release()


# --- INPUT SCHEMAS ---
class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read relative to workspace.")

class EditFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to create or overwrite.")
    content: str = Field(..., description="Full text content to write into the file.")

class RunShellInput(BaseModel):
    command: str = Field(..., description="Shell command string to execute.")

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string.")

class ListDirInput(BaseModel):
    dir_path: str = Field(default=".", description="Directory path to list files from.")

class FetchWebpageInput(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch content from.")

class DownloadWebFileInput(BaseModel):
    url: str = Field(..., description="URL of file to download.")
    save_path: str = Field(..., description="Destination file path inside the workspace.")

class ReadSqliteHistoryInput(BaseModel):
    limit: int = Field(
        default=DEFAULT_HISTORY_READ_LIMIT,
        description="Number of recent chat history messages to read.",
    )

class DeleteSqliteHistoryInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Specific session ID to delete, or leave empty/all to wipe.")

class CreateSkillInput(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_NAME_CHARS,
        description="Short skill title or slug, for example 'Python testing'.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_DESCRIPTION_CHARS,
        description="When this skill should be used and what it specializes in.",
    )
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_INSTRUCTION_CHARS,
        description="Markdown instructions for the skill, based only on the user's request and relevant context.",
    )

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


@tool(args_schema=ReadFileInput)
def read_local_file(file_path: str) -> str:
    """Read and return content from a file inside the sandboxed workspace directory with stream-based chunking and size validation to protect memory overhead."""
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.exists():
            return f"Error: File '{file_path}' not found."
        
        max_size = MAX_READ_FILE_BYTES
        file_size = target.stat().st_size
        if file_size > max_size:
            return (
                f"Error: File '{file_path}' exceeds the maximum allowed size of "
                f"{max_size} bytes."
            )

        # Quick binary check on the first chunk
        with open(target, "rb") as f:
            header_bytes = f.read(BINARY_PROBE_BYTES)
            if b'\x00' in header_bytes:
                return f"Error: File '{file_path}' appears to be a binary file and cannot be read as text."

        # Stream-based chunking read to protect memory overhead
        chunks = []
        chunk_size = FILE_READ_CHUNK_BYTES
        with open(target, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        raw_bytes = b"".join(chunks)
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool(args_schema=EditFileInput)
def edit_local_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with given content safely within workspace root, creating an HITL snapshot backup first."""
    try:
        target = SandboxManager.validate_path(file_path)
        create_hitl_snapshot(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: File '{file_path}' written securely (HITL checkpoint snapshot created)."
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool(args_schema=RunShellInput)
def run_shell_command(command: str) -> str:
    """Run an approved command inside the configured OS-isolated workspace."""
    import subprocess

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
        console.print(
            "[yellow]Commands run in a Linux OS sandbox with network disabled, "
            f"workspace-only writes, and resource limits.{risk_notice}[/yellow]"
        )
        if not _tool_invocation_approved.get():
            approval = console.input(
                f"[yellow]Approve command {' '.join(shlex.quote(arg) for arg in command_args)}? \\[y/N]: [/yellow]"
            ).strip().lower()
            if approval != "y":
                return "Error: Command execution was not approved; no command was run."
        from ..code_tasks import ACTIVE_CODE_TASK
        if ACTIVE_CODE_TASK is not None and command_args[0] == "git":
            return (
                "Error: Use checkpoint_code_task for local code-task checkpoints; "
                "manual Git commands are disabled."
            )
        from ..code_tasks import _isolated_command
        workspace_root = (
            ACTIVE_CODE_TASK.root
            if ACTIVE_CODE_TASK is not None
            else SandboxManager.root_dir
        )
        isolated_command, environment = _isolated_command(
            command_args, workspace_root
        )
        result = subprocess.run(
            isolated_command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=SHELL_COMMAND_TIMEOUT,
            env=environment,
        )
        output = result.stdout + result.stderr
        return f"Exit Code: {result.returncode}\nOutput:\n{output}"
    except RuntimeError as e:
        return f"Execution blocked: {str(e)}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

def _search_duckduckgo(query: str, client_factory=public_only_sync_client) -> str:
    from bs4 import BeautifulSoup

    if not query.strip() or len(query) > MAX_SEARCH_QUERY_CHARS:
        raise ValueError(
            f"Search query must contain 1 to {MAX_SEARCH_QUERY_CHARS} characters."
        )
    acquired = _network_slots.acquire(timeout=NETWORK_REQUEST_TIMEOUT)
    if not acquired:
        raise TimeoutError("Outbound request concurrency limit wait timed out.")
    try:
        current_url = validate_outbound_url("https://html.duckduckgo.com/html/")
        with client_factory(
            headers={"User-Agent": "PrivateAgent/1.0"},
            timeout=httpx.Timeout(NETWORK_REQUEST_TIMEOUT),
            track=False,
        ) as client:
            for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
                with client.stream(
                    "GET",
                    current_url,
                    params={"q": query} if redirect_count == 0 else None,
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_HTTP_REDIRECTS:
                            raise ValueError("HTTP redirect limit exceeded or redirect target missing.")
                        current_url = validate_outbound_url(
                            urllib.parse.urljoin(current_url, location)
                        )
                        continue
                    response.raise_for_status()
                    declared_size = response.headers.get("content-length")
                    if declared_size and int(declared_size) > MAX_WEBPAGE_BYTES:
                        raise ValueError(
                            f"Response exceeds the {MAX_WEBPAGE_BYTES}-byte size limit."
                        )
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_WEBPAGE_BYTES:
                            raise ValueError(
                                f"Response exceeds the {MAX_WEBPAGE_BYTES}-byte size limit."
                            )
                    soup = BeautifulSoup(
                        bytes(body).decode(response.encoding or "utf-8", errors="replace"),
                        "html.parser",
                    )
                    results = []
                    for result in soup.select(".result")[:MAX_SEARCH_RESULTS]:
                        anchor = result.select_one("a.result__a")
                        if not anchor or not anchor.get("href"):
                            continue
                        snippet = result.select_one(".result__snippet")
                        results.append(
                            f"Title: {anchor.get_text(' ', strip=True)}\n"
                            f"Snippet: {snippet.get_text(' ', strip=True) if snippet else ''}\n"
                            f"URL: {anchor['href']}"
                        )
                    return "\n---\n".join(results) if results else "No web search results found."
        raise ValueError("HTTP redirect limit exceeded.")
    finally:
        _network_slots.release()


@tool(args_schema=WebSearchInput)
@require_internet
def web_search(query: str) -> str:
    """Search the web for up-to-date info, technical documentation, or code references."""
    try:
        return _search_duckduckgo(query)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"

@tool(args_schema=ListDirInput)
def list_directory(dir_path: str = ".") -> str:
    """List files and subdirectories cleanly within the workspace root."""
    try:
        target = SandboxManager.validate_path(dir_path)
        if not target.is_dir():
            return f"Error: '{dir_path}' is not a valid directory."
        items = []
        for child in target.iterdir():
            if child.name.startswith('.'):
                continue
            prefix = "[DIR]  " if child.is_dir() else "[FILE] "
            rel_path = child.relative_to(SandboxManager.root_dir)
            items.append(f"{prefix} {rel_path}")
        return "\n".join(sorted(items)) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool(args_schema=FetchWebpageInput)
@require_internet
def fetch_webpage(url: str) -> str:
    """Fetch, clean, and read the full text content of a documentation or article URL using asynchronous request handlers."""
    async def _async_fetch():
        from bs4 import BeautifulSoup
        async with _public_only_async_client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=NETWORK_REQUEST_TIMEOUT,
        ) as client:
            body, encoding = await _get_limited_response(
                client,
                url,
                MAX_WEBPAGE_BYTES,
            )
        soup = BeautifulSoup(body.decode(encoding, errors="replace"), "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return (
            text[:MAX_FETCHED_WEBPAGE_CHARS] + "\n[Truncated...]"
            if len(text) > MAX_FETCHED_WEBPAGE_CHARS
            else text
        )

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _async_fetch()).result()
        else:
            return asyncio.run(_async_fetch())
    except ImportError:
        return "Failed to fetch webpage: Required package 'httpx' or 'beautifulsoup4' is not installed."
    except Exception as e:
        return f"Failed to fetch webpage: {str(e)}"

@tool(args_schema=DownloadWebFileInput)
def download_web_file(url: str, save_path: str) -> str:
    """Download any file from a web URL securely inside the workspace using asynchronous request handlers."""
    try:
        target = SandboxManager.validate_path(save_path)
    except Exception as e:
        return f"Failed to download file: {e}"
    if target.suffix.lower() in [".sh", ".exe", ".bat", ".cmd", ".msi"]:
        return f"Error: Downloading executable/script extension '{target.suffix}' is restricted by security policy."
    if not has_internet_connection():
        return "Error: Network unavailable. Please check your internet connection and try again."

    async def _async_download():
        target = SandboxManager.validate_path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        temporary_path = None
        try:
            async with _public_only_async_client(
                headers=headers,
                timeout=NETWORK_REQUEST_TIMEOUT,
            ) as client:
                body, _ = await _get_limited_response(
                    client,
                    url,
                    MAX_DOWNLOAD_BYTES,
                )
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = output.name
                output.write(body)
            os.replace(temporary_path, target)
        except Exception:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
            raise
        return f"Success: File downloaded and saved to '{save_path}'."

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _async_download()).result()
        else:
            return asyncio.run(_async_download())
    except Exception as e:
        return f"Failed to download file: {str(e)}"

@tool(args_schema=ReadSqliteHistoryInput)
def read_chat_history_from_sqlite(limit: int = DEFAULT_HISTORY_READ_LIMIT) -> str:
    """Read recent conversation log entries directly from the SQLite persistent memory database using thread-safe context management and locks."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(ACTIVE_DB_PATH, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT session_id, role, content, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
        if not rows:
            return "SQLite memory chat history is empty."
        formatted = []
        for session_id, role, content, timestamp in reversed(rows):
            formatted.append(f"[{timestamp}] Session: {session_id} | {role}: {content}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error reading SQLite history: {str(e)}"

@tool(args_schema=DeleteSqliteHistoryInput)
def delete_chat_history_from_sqlite(session_id: Optional[str] = None) -> str:
    """Delete chat history logs from the SQLite persistent database using thread-safe context management and locks."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(ACTIVE_DB_PATH, check_same_thread=False) as conn:
                if session_id:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
                    rows_affected = cursor.rowcount
                    return f"Success: Deleted {rows_affected} messages for session '{session_id}' from SQLite memory."
                else:
                    conn.execute("DELETE FROM chat_history")
                    return "Success: Wiped all chat history rows from SQLite persistent database."
    except Exception as e:
        return f"Error deleting SQLite history: {str(e)}"


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

async def load_configured_mcp_tools(server_name: str, server_config: dict) -> list:
    """Load LangChain-callable tools from one explicitly configured MCP server."""
    if not isinstance(server_config, dict):
        raise ValueError(f"MCP server '{server_name}' configuration must be an object.")
    transport = server_config.get("transport", "stdio")
    if not isinstance(transport, str) or transport not in {
        "stdio",
        "sse",
        "streamable_http",
        "websocket",
    }:
        raise ValueError(
            f"MCP server '{server_name}' has unsupported transport '{transport}'."
        )
    if transport == "stdio":
        if not isinstance(server_config.get("command"), str) or not server_config["command"].strip():
            raise ValueError(f"MCP stdio server '{server_name}' requires a command.")
        if not isinstance(server_config.get("args", []), list) or any(
            not isinstance(arg, str) for arg in server_config.get("args", [])
        ):
            raise ValueError(f"MCP stdio server '{server_name}' args must be a string list.")
        environment = server_config.get("env")
        if environment is not None and (
            not isinstance(environment, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in environment.items())
        ):
            raise ValueError(f"MCP stdio server '{server_name}' env must contain string values.")
    normalized_config = dict(server_config)
    if transport in {"sse", "streamable_http", "websocket"}:
        url = server_config.get("url")
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        supported_schemes = (
            {"ws", "wss"} if transport == "websocket" else {"http", "https"}
        )
        if (
            parsed is None
            or parsed.scheme not in supported_schemes
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError(
                f"MCP {transport} server '{server_name}' requires a valid URL "
                f"using {', '.join(sorted(supported_schemes))}."
            )
    for timeout_key in ("timeout", "sse_read_timeout"):
        if timeout_key in server_config:
            if isinstance(server_config[timeout_key], bool):
                raise ValueError(
                    f"MCP server '{server_name}' {timeout_key} must be a number."
                )
            try:
                timeout_value = float(server_config[timeout_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MCP server '{server_name}' {timeout_key} must be a number."
                ) from exc
            if not 0.1 <= timeout_value <= 120:
                raise ValueError(
                    f"MCP server '{server_name}' {timeout_key} must be between "
                    "0.1 and 120 seconds."
                )
    if transport in {"sse", "streamable_http", "websocket"}:
        try:
            normalized_config["url"] = _validate_explicit_service_url(
                server_config["url"], supported_schemes
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"MCP server '{server_name}' endpoint was rejected: {exc}"
            ) from exc
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "MCP servers are configured, but langchain-mcp-adapters is not installed."
        ) from exc

    sandbox_dir = None
    if transport in {"sse", "streamable_http"}:
        normalized_config["httpx_client_factory"] = _mcp_http_client_factory
    elif transport == "stdio":
        from ..code_tasks import _isolated_command

        sandbox_dir = tempfile.mkdtemp(prefix="private-agent-mcp-")
        _mcp_sandbox_dirs.append(sandbox_dir)
        try:
            command = [
                server_config["command"],
                *server_config.get("args", []),
            ]
            isolated_command, sandbox_environment = _isolated_command(
                command, pathlib.Path(sandbox_dir)
            )
        except Exception:
            _remove_mcp_sandbox_dir(sandbox_dir)
            raise
        normalized_config["command"] = isolated_command[0]
        normalized_config["args"] = isolated_command[1:]
        sandbox_environment.update(server_config.get("env") or {})
        normalized_config["env"] = sandbox_environment
        normalized_config.pop("cwd", None)
    client = MultiServerMCPClient({server_name: normalized_config})
    try:
        server_tools = await client.get_tools(server_name=server_name)
    except Exception as exc:
        if sandbox_dir is not None:
            _remove_mcp_sandbox_dir(sandbox_dir)
        raise RuntimeError(f"Failed to load MCP server '{server_name}': {exc}") from exc
    if sandbox_dir is not None and not server_tools:
        _remove_mcp_sandbox_dir(sandbox_dir)
    return list(server_tools)


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
