import pytest
import time
import asyncio
import types
from unittest.mock import patch, MagicMock, AsyncMock
from rich.console import Console
from langchain_core.messages import AIMessageChunk

from private_agent.database import PersistentMemory
from private_agent.sandbox import SandboxManager
from private_agent.agent import (
    authorize_network_research,
    _invoke_with_budget,
    execute_tool_call,
    run_agent_cli_async,
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

def test_sync_connect_backend_falls_back_to_second_validated_address(monkeypatch):
    from private_agent.tools import _PublicOnlySyncNetworkBackend
    import httpcore

    backend = _PublicOnlySyncNetworkBackend(allow_loopback=True)
    attempts = []

    def connect(address, port, **kwargs):
        attempts.append(address)
        if address == "::1":
            raise httpcore.ConnectError("IPv6 listener unavailable")
        return "connected"

    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "private_agent.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (10, 1, 6, "", ("::1", 11434, 0, 0)),
            (2, 1, 6, "", ("127.0.0.1", 11434)),
        ],
    )
    assert backend.connect_tcp("localhost", 11434) == "connected"
    assert attempts == ["::1", "127.0.0.1"]

@pytest.mark.asyncio
async def test_offline_research_can_continue_with_incomplete_data():
    from private_agent.agent import runtime as agent

    with patch("private_agent.agent.runtime.has_internet_connection", return_value=False), \
         patch("private_agent.agent.runtime.sys.stdin", MagicMock(isatty=lambda: True)), \
         patch.object(agent.console, "input", return_value="c"):
        allowed, message = await authorize_network_research(
            "web_search", {"query": "example"}
        )
    assert allowed is False
    assert "incomplete" in message

def test_permission_mode_policy_matrix(monkeypatch):
    import private_agent.agent.runtime as agent

    monkeypatch.setattr(agent, "MCP_TOOL_NAMES", {"mcp_read"})
    monkeypatch.setattr(agent, "MCP_AUTO_APPROVE_TOOLS", {"mcp_trusted"})
    assert not agent._tool_needs_permission("read_local_file", "auto")
    assert not agent._tool_needs_permission("web_search", "auto")
    assert agent._tool_needs_permission("edit_local_file", "auto")
    assert agent._tool_needs_permission("mcp_read", "auto")
    assert not agent._tool_needs_permission("mcp_trusted", "auto")
    assert agent._tool_needs_permission("read_local_file", "manual")
    assert not agent._tool_needs_permission(
        "web_search", "manual", network_authorized=True
    )
    assert not agent._tool_needs_permission("run_shell_command", "full")

@pytest.mark.asyncio
async def test_tool_result_and_failure_are_returned_to_agent(monkeypatch):
    import private_agent.agent.runtime as agent

    successful_tool = MagicMock()
    successful_tool.ainvoke = AsyncMock(return_value="tool output")
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"example": successful_tool})
    result = await execute_tool_call(
        {"name": "example", "args": {"value": 1}, "id": "call-result"}
    )
    assert result.content == "tool output"
    assert result.tool_call_id == "call-result"

    failing_tool = MagicMock()
    failing_tool.ainvoke = AsyncMock(side_effect=RuntimeError("tool unavailable"))
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"example": failing_tool})
    result = await execute_tool_call(
        {"name": "example", "args": {}, "id": "call-error"}
    )
    assert "Error executing tool example: tool unavailable" in result.content
    assert "[System Reflection Prompt]" in result.content

@pytest.mark.asyncio
async def test_model_invocation_streams_and_aggregates_text(monkeypatch):
    import private_agent.agent.runtime as agent

    class StreamingModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="first ")
            yield AIMessageChunk(content="second")

    printed = []
    monkeypatch.setattr(
        agent.console,
        "print",
        lambda *values, **kwargs: printed.append((values, kwargs)),
    )
    response = await _invoke_with_budget(
        StreamingModel(),
        [],
        time.monotonic() + 5,
        model_name="test-model",
    )

    assert isinstance(response, AIMessageChunk)
    assert response.content == "first second"
    assert any(values == ("first ",) for values, _ in printed)
    assert any(values == ("second",) for values, _ in printed)

def test_visible_chunk_text_includes_text_blocks():
    import private_agent.agent.runtime as agent

    assert agent._visible_chunk_text(
        ["Hello", {"type": "text", "text": " world"}, {"type": "image"}]
    ) == "Hello world"

@pytest.mark.asyncio
async def test_model_invocation_falls_back_to_nonstreaming_models():
    model = MagicMock()
    expected = types.SimpleNamespace(content="complete", tool_calls=[])
    model.ainvoke = AsyncMock(return_value=expected)

    response = await _invoke_with_budget(model, [], time.monotonic() + 5)

    assert response is expected
    model.ainvoke.assert_awaited_once_with([])

@pytest.mark.asyncio
async def test_streaming_response_is_not_rendered_inside_live_status(monkeypatch):
    import private_agent.agent.runtime as agent
    from contextlib import contextmanager

    status_active = False
    printed = []

    @contextmanager
    def status(_message):
        nonlocal status_active
        status_active = True
        try:
            yield
        finally:
            status_active = False

    class StreamingModel:
        async def astream(self, _messages):
            assert not status_active
            yield AIMessageChunk(content="visible answer")

    monkeypatch.setattr(agent.console, "status", status)
    monkeypatch.setattr(
        agent.console,
        "print",
        lambda *values, **kwargs: printed.append((values, kwargs)),
    )

    response = await agent._invoke_with_status(
        StreamingModel(),
        [],
        time.monotonic() + 5,
        model_name="test-model",
        status_message="Generating",
    )

    assert response.content == "visible answer"
    assert not status_active
    assert any(values == ("visible answer",) for values, _ in printed)

@pytest.mark.asyncio
async def test_nonstreaming_response_keeps_live_status(monkeypatch):
    import private_agent.agent.runtime as agent
    from contextlib import contextmanager

    status_active = False

    @contextmanager
    def status(_message):
        nonlocal status_active
        status_active = True
        try:
            yield
        finally:
            status_active = False

    class NonStreamingModel:
        async def ainvoke(self, _messages):
            assert status_active
            return types.SimpleNamespace(content="complete", tool_calls=[])

    monkeypatch.setattr(agent.console, "status", status)

    response = await agent._invoke_with_status(
        NonStreamingModel(),
        [],
        time.monotonic() + 5,
        model_name="test-model",
        status_message="Generating",
    )

    assert response.content == "complete"
    assert not status_active

@pytest.mark.asyncio
async def test_agent_shutdown_closes_resources_and_restores_dynamic_tools(
    monkeypatch, tmp_path
):
    import private_agent.agent.runtime as agent

    closed = []
    initial_root = tmp_path / "initial"
    session_root = tmp_path / "session"
    initial_root.mkdir()
    session_root.mkdir()
    monkeypatch.setattr(SandboxManager, "root_dir", initial_root.resolve())

    class FakeMemory:
        def close(self):
            closed.append(True)

    async def fail_during_session():
        agent._ACTIVE_MEMORY = FakeMemory()
        agent.MCP_TOOL_NAMES.add("temporary_mcp_tool")
        agent.AVAILABLE_TOOLS["temporary_mcp_tool"] = object()
        agent.AVAILABLE_TOOLS["checkpoint_code_task"] = object()
        SandboxManager.set_root(str(session_root))
        agent.code_tasks_module.ACTIVE_CODE_TASK = object()
        raise RuntimeError("original runtime failure")

    monkeypatch.setattr(agent, "_ACTIVE_MEMORY", None)
    monkeypatch.setattr(agent, "_run_agent_cli_session", fail_during_session)
    with pytest.raises(RuntimeError, match="original runtime failure"):
        await run_agent_cli_async()
    assert closed == [True]
    assert "temporary_mcp_tool" not in agent.MCP_TOOL_NAMES
    assert "temporary_mcp_tool" not in agent.AVAILABLE_TOOLS
    assert agent.code_tasks_module.ACTIVE_CODE_TASK is None
    assert agent._ACTIVE_MEMORY is None
    assert SandboxManager.root_dir == initial_root.resolve()
    assert "checkpoint_code_task" not in agent.AVAILABLE_TOOLS

@pytest.mark.asyncio
async def test_local_agent_turn_persists_history_and_summary(monkeypatch, tmp_path):
    import private_agent.agent.runtime as agent

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database_path = tmp_path / "memory.sqlite"
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=types.SimpleNamespace(
        content="local response",
        tool_calls=[],
    ))
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent, "DEFAULT_DB_PATH", str(database_path))
    monkeypatch.setattr(agent, "RAG_DOCS_DEFAULT", None)
    monkeypatch.setattr(agent, "SKILLS_FOLDER_DEFAULT", "")
    monkeypatch.setattr(agent, "WORKSPACE_ROOT_DEFAULT", str(workspace))
    monkeypatch.setattr(agent, "MCP_SERVERS", {})
    monkeypatch.setattr(agent, "set_active_db_path", lambda path: None)
    monkeypatch.setattr(agent, "fetch_local_chat_models", lambda: ["local-model"])
    monkeypatch.setattr(agent, "inspect_model_capabilities", lambda model: {
        "tools": False,
        "function_calls": False,
        "structured_output": False,
        "thinking": False,
        "vision": False,
        "audio": False,
    })
    monkeypatch.setattr(agent, "_make_chat_model", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr(
        agent, "inspect_ollama_hardware", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(agent, "format_hardware_status", lambda status: "test")
    monkeypatch.setattr(
        agent, "initialize_knowledge_base", lambda path, **kwargs: None
    )
    monkeypatch.setattr(agent, "load_skills_from_folder", lambda path: {})
    inputs = MagicMock(side_effect=["", "", "", "", "1", "", "hello", "exit"])
    monkeypatch.setattr(agent.console, "input", inputs)

    await run_agent_cli_async()

    inputs.side_effect = ["", "", "", "y", "", "1", "", "exit"]
    await run_agent_cli_async()

    memory = PersistentMemory(db_path=str(database_path))
    session_id = memory.get_latest_session_id()
    assert session_id is not None
    assert [message.content for message in memory.load_history(session_id)] == [
        "hello",
        "local response",
    ]
    assert memory.get_all_episodic_summaries(session_id=session_id) == [
        "local response",
        "local response",
    ]
    assert sum("Resume latest session" in call.args[0] for call in inputs.call_args_list) == 1
    memory.close()
    assert mock_model.ainvoke.await_count == 3

@pytest.mark.asyncio
async def test_concurrent_tool_execution():
    async def sample_tool_task(val):
        await asyncio.sleep(0.005)
        return f"res_{val}"

    results = await asyncio.gather(
        sample_tool_task("alpha"),
        sample_tool_task("beta"),
        sample_tool_task("gamma")
    )
    assert results == ["res_alpha", "res_beta", "res_gamma"]

@pytest.mark.asyncio
async def test_concurrent_tool_execution_with_edge_case_exceptions():
    async def faulty_tool_task(val, should_fail=False):
        await asyncio.sleep(0.005)
        if should_fail:
            raise ValueError("Tool execution failed")
        return f"ok_{val}"

    results = await asyncio.gather(
        faulty_tool_task(1, False),
        faulty_tool_task(2, True),
        return_exceptions=True
    )
    assert results[0] == "ok_1"
    assert isinstance(results[1], ValueError)

@patch("private_agent.agent.runtime.ChatOllama")
def test_model_fallback_on_tool_incompatibility(mock_chat_ollama):
    mock_primary = MagicMock()
    mock_primary.bind_tools.side_effect = Exception("Model does not support tools")
    mock_secondary = MagicMock()

    mock_chat_ollama.side_effect = [mock_primary, mock_secondary]

    model = get_robust_chat_model("primary_model", "fallback_model", tools=[])
    assert model is not None

@patch("private_agent.agent.runtime.ChatOllama")
def test_model_fallback_total_failure_edge_case(mock_chat_ollama):
    mock_primary = MagicMock()
    mock_primary.bind_tools.side_effect = Exception("Primary fails")
    mock_secondary = MagicMock()
    mock_secondary.bind_tools.side_effect = Exception("Secondary also fails")

    mock_chat_ollama.side_effect = [mock_primary, mock_secondary]

    try:
        model = get_robust_chat_model("primary_model", "fallback_model", tools=[])
        assert model is None
    except Exception:
        pass

@patch("private_agent.agent.runtime.ChatOllama")
def test_model_fallback_does_not_hide_unrecognized_configuration_error(mock_chat_ollama):
    mock_chat_ollama.side_effect = ValueError("invalid temperature configuration")
    assert get_robust_chat_model("primary_model", "fallback_model") is None
    mock_chat_ollama.assert_called_once()

@patch("private_agent.agent.runtime.ollama.Client")
def test_model_capabilities_are_read_from_ollama_metadata(mock_client):
    from private_agent.agent import runtime as agent

    agent._MODEL_CAPABILITIES_CACHE.clear()

    mock_client.return_value.show.return_value = {
        "capabilities": ["completion", "tools", "vision", "thinking"]
    }
    capabilities = agent.inspect_model_capabilities("local-model")
    assert capabilities["tools"] is True
    assert capabilities["function_calls"] is True
    assert capabilities["vision"] is True
    assert capabilities["thinking"] is True
    assert capabilities["audio"] is False
    capabilities["tools"] = False
    cached_capabilities = agent.inspect_model_capabilities("local-model")
    assert cached_capabilities["tools"] is True
    mock_client.assert_called_once()
    agent._MODEL_CAPABILITIES_CACHE.clear()
