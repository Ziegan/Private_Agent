import pytest
import asyncio
import sys
import types
from unittest.mock import patch, MagicMock, AsyncMock
from rich.console import Console

from private_agent.sandbox import SandboxManager, evaluate_shell_command
from private_agent.tools import (
    read_local_file,
    edit_local_file,
    run_shell_command,
    download_web_file,
    load_mcp_tools,
    load_configured_mcp_tools,
    describe_tool_catalog,
    get_local_datetime,
    close_outbound_http_clients,
    close_mcp_sandbox_dirs,
    ollama_client_kwargs,
    ollama_langchain_client_kwargs,
)
from private_agent.agent import (
    execute_tool_call,
)
try:
    from private_agent.agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in private_agent.agent
    def get_robust_chat_model(primary_model_name, fallback_model_name, tools=None):
        from langchain_ollama import ChatOllama
        try:
            model = ChatOllama(model=primary_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model
        except Exception:
            model = ChatOllama(model=fallback_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model


console = Console()

@pytest.mark.asyncio
async def test_mcp_approval_redacts_sensitive_arguments(monkeypatch):
    import private_agent.agent.runtime as agent

    mcp_tool = MagicMock()
    mcp_tool.name = "mcp_secret_test"
    mcp_tool.description = "Call the remote service"
    mcp_tool.ainvoke = AsyncMock(return_value="done")
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"mcp_secret_test": mcp_tool})
    monkeypatch.setattr(agent, "MCP_TOOL_NAMES", {"mcp_secret_test"})
    monkeypatch.setattr(agent, "MCP_AUTO_APPROVE_TOOLS", set())
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    logged = []
    prompts = []
    monkeypatch.setattr(agent.console, "print", lambda *args, **kwargs: logged.append(str(args)))
    monkeypatch.setattr(
        agent.console,
        "input",
        lambda prompt: prompts.append(str(prompt)) or "y",
    )
    call_args = {
        "api_key": "secret-key-value",
        "nested": {"access_token": "secret-token-value"},
        "query": "ordinary query",
    }
    result = await execute_tool_call({
        "name": "mcp_secret_test",
        "args": call_args,
        "id": "mcp-call",
    })
    assert result.content == "done"
    mcp_tool.ainvoke.assert_awaited_once_with(call_args)
    assert all(
        secret not in line
        for line in logged + prompts
        for secret in ("secret-key-value", "secret-token-value")
    )
    assert "[REDACTED]" in prompts[0]

def test_sandbox_path_validation(temp_workspace):
    valid_path = SandboxManager.validate_path("test.txt")
    assert valid_path == (temp_workspace / "test.txt").resolve()

    with pytest.raises(PermissionError):
        SandboxManager.validate_path("../outside.txt")

    with pytest.raises(PermissionError):
        SandboxManager.validate_path("/etc/passwd")

def test_shell_command_evaluation():
    assert evaluate_shell_command("ls -la") is False
    assert evaluate_shell_command("python -m pytest") is False

    assert evaluate_shell_command("sudo apt update") is True
    assert evaluate_shell_command("rm -rf /") is True
    assert evaluate_shell_command("chmod 777 script.py") is True

def test_file_tools_in_sandbox(temp_workspace):
    file_path = "hello.txt"
    content = "Hello, Private Agent Sandbox!"

    write_res = edit_local_file.invoke({"file_path": file_path, "content": content})
    assert "Success" in write_res

    read_res = read_local_file.invoke({"file_path": file_path})
    assert read_res == content

def test_read_binary_file_guard(temp_workspace):
    bin_file = temp_workspace / "binary.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03" * 500)

    res = read_local_file.invoke({"file_path": "binary.bin"})
    assert "Error" in res
    assert "binary file" in res

def test_read_oversized_file_guard(temp_workspace):
    large_file = temp_workspace / "large.txt"
    large_file.write_text("A" * (1024 * 1024 + 10), encoding="utf-8")

    res = read_local_file.invoke({"file_path": "large.txt"})
    assert "Error" in res
    assert "exceeds the maximum allowed size" in res

def test_non_interactive_shell_blocking(temp_workspace):
    res = run_shell_command.invoke({"command": "sudo rm -rf /"})
    assert "Error" in res
    assert "requires an interactive terminal" in res

def test_shell_command_requires_explicit_approval(temp_workspace):
    with patch("private_agent.tools.shell.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "private_agent.tools.shell.console.input", return_value="n"
    ), patch("subprocess.run") as run:
        result = run_shell_command.invoke({"command": "echo no"})
    assert "was not approved" in result
    run.assert_not_called()

def test_tool_catalog_shows_origin_permissions_and_effects(monkeypatch):
    import private_agent.tools as tools

    monkeypatch.setattr(tools, "AVAILABLE_TOOLS", {
        "edit_local_file": MagicMock(description="Write a workspace file"),
        "web_search": MagicMock(description="Search the web"),
        "mcp_read": MagicMock(description="Read from a server"),
    })
    monkeypatch.setattr(tools, "MCP_TOOL_NAMES", {"mcp_read"})
    monkeypatch.setattr(tools, "MCP_TOOL_SOURCES", {"mcp_read": "docs"})
    monkeypatch.setattr(tools, "MCP_TOOL_TRANSPORTS", {"mcp_read": "streamable_http"})
    monkeypatch.setattr(tools, "MCP_AUTO_APPROVE_TOOLS", set())
    catalog = {entry["name"]: entry for entry in describe_tool_catalog()}
    assert catalog["mcp_read"]["origin"] == "docs"
    assert catalog["mcp_read"]["permission"] == "approval required"
    assert catalog["mcp_read"]["effects"] == "network-capable; server-defined effects"
    assert catalog["web_search"]["effects"] == "network"
    assert catalog["edit_local_file"]["effects"] == "may modify local state"

def test_local_datetime_tool_reports_local_calendar_and_timezone():
    from datetime import datetime

    result = get_local_datetime.invoke({})
    local_now = datetime.now().astimezone()

    assert f"Weekday: {local_now:%A}" in result
    assert f"Month: {local_now:%B}" in result
    assert f"Year: {local_now:%Y}" in result
    assert f"UTC{local_now:%z}" in result
    assert "Local date:" in result
    assert "Local time:" in result

def test_local_datetime_tool_is_listed_as_read_only():
    catalog = {entry["name"]: entry for entry in describe_tool_catalog()}

    assert catalog["get_local_datetime"]["permission"] == (
        "no approval required; reads local system clock"
    )
    assert catalog["get_local_datetime"]["effects"] == "read-only/local"

def test_agent_current_datetime_context_includes_local_date_and_timezone():
    from datetime import datetime
    from private_agent.agent import _current_datetime_context

    context = _current_datetime_context()
    now = datetime.now().astimezone()

    assert f"{now:%A, %B} {now.day}, {now.year}" in context
    assert f"{now:%H:%M:%S %Z} (UTC{now:%z})" in context
    assert "authoritative current date/time" in context

def test_tool_catalog_prints_each_entry_on_one_line():
    import io
    from rich.console import Console as RichConsole
    from private_agent.agent import _print_tool_catalog

    output = io.StringIO()
    console = RichConsole(file=output, width=56, force_terminal=False)
    entries = [
        {
            "name": "one",
            "origin": "built-in",
            "permission": "approval required",
            "effects": "read-only",
            "description": "A deliberately long tool description that needs truncation.",
        },
        {
            "name": "two",
            "origin": "MCP",
            "permission": "approval required",
            "effects": "network",
            "description": "Another deliberately long tool description.",
        },
    ]
    with patch("private_agent.agent.runtime.console", console):
        _print_tool_catalog(entries)
    lines = output.getvalue().splitlines()
    assert len(lines) == len(entries) + 1
    assert all(len(line) <= console.width for line in lines)

@pytest.mark.asyncio
async def test_mcp_transport_schema_validation(monkeypatch):
    with pytest.raises(ValueError, match="requires a command"):
        await load_configured_mcp_tools("bad-stdio", {"transport": "stdio"})
    with pytest.raises(ValueError, match="valid URL"):
        await load_configured_mcp_tools(
            "bad-http",
            {"transport": "streamable_http", "url": "https://user:secret@example.test"},
        )
    with pytest.raises(ValueError, match="between 0.1 and 120"):
        await load_configured_mcp_tools(
            "bad-timeout",
            {
                "transport": "streamable_http",
                "url": "https://example.test/mcp",
                "timeout": 1000,
            },
        )

def test_ollama_client_kwargs_use_pinned_transports():
    sync = ollama_client_kwargs()
    langchain = ollama_langchain_client_kwargs()
    assert "client" not in sync
    assert sync["transport"]._pool._network_backend._allow_loopback is True
    assert langchain["sync_client_kwargs"]["transport"]._pool._network_backend._allow_loopback
    assert langchain["async_client_kwargs"]["transport"]._pool._network_backend._allow_loopback
    assert sync["trust_env"] is False
    assert sync["follow_redirects"] is False

def test_langchain_ollama_clients_are_tracked_and_closed():
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    import private_agent.tools as tools

    chat = ChatOllama(
        model="test",
        base_url="http://localhost:11434",
        **ollama_langchain_client_kwargs(),
    )
    embeddings = OllamaEmbeddings(
        model="test",
        base_url="http://localhost:11434",
        **ollama_langchain_client_kwargs(),
    )
    tools.track_ollama_http_clients(chat)
    tools.track_ollama_http_clients(embeddings)
    expected_clients = {
        chat._client._client,
        chat._async_client._client,
        embeddings._client._client,
        embeddings._async_client._client,
    }
    assert expected_clients.issubset(set(tools._outbound_http_clients))
    asyncio.run(close_outbound_http_clients())
    assert all(client.is_closed for client in expected_clients)

def test_mcp_stdio_tools_are_loaded_with_isolated_command(monkeypatch):
    captured = {}

    class FakeMCPClient:
        def __init__(self, configs):
            captured["configs"] = configs

        async def get_tools(self, server_name):
            captured["server_name"] = server_name
            return []

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FakeMCPClient
    isolated = ["/usr/bin/bwrap", "--unshare-all", "--", "python", "-m", "server"]
    monkeypatch.setattr(
        "private_agent.code_tasks._isolated_command",
        lambda command, root: (isolated, {}),
    )
    try:
        with patch.dict(
            sys.modules,
            {
                "langchain_mcp_adapters": parent,
                "langchain_mcp_adapters.client": adapter,
            },
        ):
            assert asyncio.run(
                load_configured_mcp_tools(
                    "local",
                    {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["-m", "server"],
                    },
                )
            ) == []
        config = captured["configs"]["local"]
        assert config["command"] == isolated[0]
        assert config["args"] == isolated[1:]
        assert "cwd" not in config
    finally:
        close_mcp_sandbox_dirs()

def test_download_restricted_extension(temp_workspace):
    res = download_web_file.invoke({"url": "http://example.com/script.sh", "save_path": "script.sh"})
    assert "Error" in res
    assert "restricted by security policy" in res

def test_shell_command_tool(temp_workspace):
    with patch("private_agent.tools.shell.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "private_agent.tools.shell.console.input", return_value="y"
    ):
        res = run_shell_command.invoke({"command": "echo 'sandbox integration test'"})
    assert "Exit Code: 0" in res
    assert "sandbox integration test" in res

@patch("mcp.client.stdio.stdio_client")
@patch("mcp.client.session.ClientSession")
@pytest.mark.asyncio
async def test_load_mcp_tools_mocked(mock_client_session, mock_stdio_client):
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = MagicMock(tools=[
        MagicMock(name="mcp_tool_1", description="MCP tool desc", inputSchema={"type": "object"})
    ])
    mock_client_session.return_value.__aenter__.return_value = mock_session

    tools = await load_mcp_tools()
    assert isinstance(tools, list)

@patch("mcp.client.stdio.stdio_client", side_effect=Exception("Connection timeout"))
@pytest.mark.asyncio
async def test_load_mcp_tools_connection_failure_edge_case(mock_stdio_client):
    tools = await load_mcp_tools()
    assert isinstance(tools, list)
    assert len(tools) == 0
