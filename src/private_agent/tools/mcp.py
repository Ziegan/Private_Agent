"""Configured MCP server validation, isolation, and tool discovery."""

import pathlib
import shutil
import tempfile
import urllib.parse

from rich.console import Console

from .network import mcp_http_client_factory, validate_explicit_service_url

console = Console()
_mcp_sandbox_dirs = []


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
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            )
        ):
            raise ValueError(
                f"MCP stdio server '{server_name}' env must contain string values."
            )
    normalized_config = dict(server_config)
    supported_schemes = (
        {"ws", "wss"} if transport == "websocket" else {"http", "https"}
    )
    if transport in {"sse", "streamable_http", "websocket"}:
        url = server_config.get("url")
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
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
            normalized_config["url"] = validate_explicit_service_url(
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
        normalized_config["httpx_client_factory"] = mcp_http_client_factory
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
