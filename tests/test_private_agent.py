import pytest
import pathlib
import tempfile
import time
import json
import asyncio
import sys
import types
import subprocess
import shutil
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel
from langchain_core.messages import HumanMessage, AIMessage

from src.config import load_or_create_config
from src.database import PersistentMemory
from src.sandbox import SandboxManager, evaluate_shell_command
from src.tools import (
    read_local_file,
    edit_local_file,
    run_shell_command,
    download_web_file,
    read_chat_history_from_sqlite,
    delete_chat_history_from_sqlite,
    set_active_db_path,
    load_mcp_tools,
    load_configured_mcp_tools,
    check_internet_connection,
    validate_outbound_url,
    _get_limited_response,
    _PublicOnlyNetworkBackend,
    describe_tool_catalog,
    close_outbound_http_clients,
    close_mcp_sandbox_dirs,
    ollama_client_kwargs,
    ollama_langchain_client_kwargs,
    create_skill,
    set_active_skill_runtime,
    _PublicOnlySyncHTTPTransport,
    _PublicOnlyAsyncHTTPTransport,
    _search_duckduckgo,
    _validate_explicit_service_url,
)
from src.agent import (
    authorize_network_research,
    execute_tool_call,
    _validate_online_base_url,
    _make_online_chat_model,
    select_online_model,
    trim_history_to_context_budget,
    _build_bounded_user_input,
    _token_count,
    format_retrieved_citations,
    run_agent_cli_async,
)
try:
    from src.agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in src.agent
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

from src.skills import load_skills_from_folder, match_skill_by_relevancy
from src.rag import (
    initialize_knowledge_base,
    _split_into_chunks,
    _read_index_state,
    _write_index_state_atomic,
)
from src.code_tasks import (
    CodeTaskWorkspace,
    is_code_task_request,
    _isolated_command,
)
from src.hardware import (
    format_hardware_status,
    inspect_ollama_hardware,
    normalize_acceleration_mode,
    ollama_acceleration_options,
)
from src.media_tools import (
    capture_webcam_image,
    list_microphone_devices,
    list_microphone_devices,
    record_microphone_audio,
    consume_captured_image,
    captured_image_message,
)

console = Console()

class SessionTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.start_time = time.time()
        self.current_test = ""

@pytest.fixture(scope="session")
def session_tracker():
    tracker = SessionTracker()
    yield tracker
    duration = time.time() - tracker.start_time
    total = tracker.passed + tracker.failed
    summary_text = (
        f"[bold]Total Tests Executed:[/bold] {total}\n"
        f"[bold green]Passed:[/bold green] {tracker.passed}\n"
        f"[bold red]Failed:[/bold red] {tracker.failed}\n"
        f"[bold cyan]Total Duration:[/bold cyan] {duration:.2f}s"
    )
    console.print(Panel(summary_text, title="Test Suite Summary Report", border_style="cyan"))

@pytest.fixture(autouse=True)
def track_test_progress(request, session_tracker):
    test_name = request.node.name
    session_tracker.current_test = test_name
    console.print(f"[yellow][RUNNING][/yellow] Executing test: [bold]{test_name}[/bold]...")
    start = time.time()
    try:
        yield
        elapsed = time.time() - start
        session_tracker.passed += 1
        console.print(f"[green][PASSED][/green] {test_name} ({elapsed:.3f}s)\n")
    except Exception as e:
        elapsed = time.time() - start
        session_tracker.failed += 1
        console.print(f"[red][FAILED][/red] {test_name} ({elapsed:.3f}s) - Error: {e}\n")
        raise

@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name
    set_active_db_path(db_path)
    yield db_path
    if pathlib.Path(db_path).exists():
        pathlib.Path(db_path).unlink()

@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_root = SandboxManager.root_dir
        SandboxManager.set_root(tmpdir)
        yield pathlib.Path(tmpdir)
        SandboxManager.set_root(str(original_root))


def test_hardware_acceleration_policy_options():
    assert normalize_acceleration_mode("auto") == "auto"
    assert normalize_acceleration_mode("CPU") == "cpu"
    assert normalize_acceleration_mode("prefer-accelerator") == "accelerator"
    assert normalize_acceleration_mode("unsupported") == "auto"
    assert ollama_acceleration_options("auto") == {}
    assert ollama_acceleration_options("accelerator") == {}
    assert ollama_acceleration_options("cpu") == {"num_gpu": 0}


def test_local_chat_model_applies_configured_acceleration_mode():
    from src import agent

    for mode, expected in (
        ("auto", {}),
        ("accelerator", {}),
        ("cpu", {"num_gpu": 0}),
    ):
        with patch("src.agent.HARDWARE_ACCELERATION_MODE", mode), patch(
            "src.agent.ChatOllama"
        ) as chat_model:
            agent._make_chat_model(
                "test-model",
                thinking_enabled=False,
                thinking_effort="medium",
                supports_thinking=False,
            )
        actual_options = chat_model.call_args.kwargs
        if expected:
            assert actual_options["num_gpu"] == expected["num_gpu"]
        else:
            assert "num_gpu" not in actual_options
        assert actual_options["sync_client_kwargs"]["transport"].__class__ is (
            _PublicOnlySyncHTTPTransport
        )
        assert actual_options["async_client_kwargs"]["transport"].__class__ is (
            _PublicOnlyAsyncHTTPTransport
        )


def test_hardware_status_reports_ollama_vram_and_server_host():
    class FakeOllama:
        def ps(self):
            return {
                "models": [
                    {
                        "name": "local-model",
                        "size": 8_000_000_000,
                        "size_vram": 5 * 1024**3,
                    }
                ]
            }

    status = inspect_ollama_hardware(
        "http://localhost:11434",
        client_factory=lambda **kwargs: FakeOllama(),
    )
    assert status["server"] == "local Ollama server (localhost)"
    assert status["placement"] == "GPU VRAM allocation reported by Ollama"
    assert status["models"][0]["size_vram"] == 5 * 1024**3
    assert "does not identify CUDA, ROCm, Vulkan" in status["detail"]
    assert "5.00 GiB VRAM" in format_hardware_status(status)


def test_hardware_status_remote_server_and_cpu_fallback_reporting():
    class FakeOllama:
        def ps(self):
            return {"models": [{"model": "remote-model", "size_vram": 0}]}

    status = inspect_ollama_hardware(
        "http://192.0.2.10:11434",
        "cpu",
        client_factory=lambda **kwargs: FakeOllama(),
    )
    assert "192.0.2.10" in status["server"]
    assert "not the CLI machine" in status["server"]
    assert status["placement"] == "Ollama reports no VRAM allocation for loaded models"
    formatted = format_hardware_status(status)
    assert "CPU requested (Ollama num_gpu=0)" in formatted
    assert "no VRAM allocation" in formatted


def test_hardware_status_handles_unavailable_ollama_server():
    def unavailable(**kwargs):
        raise ConnectionError("connection refused")

    status = inspect_ollama_hardware(
        "http://localhost:11434",
        client_factory=unavailable,
    )
    assert status["placement"] == "unknown"
    assert "Could not query Ollama runtime status" in status["error"]


def test_rag_embeddings_receive_cpu_acceleration_policy(tmp_path, monkeypatch):
    from src import rag

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("Local hardware policy test.", encoding="utf-8")
    monkeypatch.setattr(rag, "HARDWARE_ACCELERATION_MODE", "cpu")
    with patch("src.rag.OllamaEmbeddings") as embeddings, patch(
        "src.rag.Chroma"
    ) as chroma:
        chroma.from_documents.return_value = MagicMock()
        assert initialize_knowledge_base(str(docs)) is not None
    assert embeddings.call_args.kwargs["num_gpu"] == 0


def test_code_task_request_detection_requires_code_intent():
    assert is_code_task_request("Create a Python application with a unit test")
    assert is_code_task_request("Please fix the bug in this script")
    assert not is_code_task_request("Explain how Python applications work")
    assert not is_code_task_request("Research local database options")


def test_code_workspace_isolated_git_and_test_gated_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Private Agent Test")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "user.email")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "private-agent-test@example.invalid")

    workspace = CodeTaskWorkspace.create(
        "Create a sample Python application",
        str(tmp_path / "projects"),
        confirm_without_git=lambda reason: False,
    )
    assert workspace is not None
    assert workspace.git_enabled
    assert workspace.root.parent == (tmp_path / "projects")
    assert pathlib.Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=workspace.root,
            text=True,
        ).strip()
    ) == workspace.root
    assert not subprocess.check_output(
        ["git", "remote"],
        cwd=workspace.root,
        text=True,
    ).strip()

    (workspace.root / "app.py").write_text("def answer():\n    return 42\n")
    (workspace.root / "tests").mkdir()
    (workspace.root / "tests" / "test_app.py").write_text(
        "from app import answer\n\n\ndef test_answer():\n    assert answer() == 41\n"
    )
    blocked = workspace.checkpoint("implementation", ["app.py"])
    assert "Include at least one relevant unit-test file" in blocked
    failed_tests = workspace.checkpoint(
        "implementation",
        ["app.py", "tests/test_app.py"],
    )
    assert "Checkpoint blocked because tests did not pass" in failed_tests
    (workspace.root / "tests" / "test_app.py").write_text(
        "from app import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
    )
    committed = workspace.checkpoint(
        "implementation",
        ["app.py", "tests/test_app.py"],
    )
    assert committed.startswith("Checkpoint committed locally:")
    assert "Tests passed:" in workspace.last_test_result

    from src import code_tasks

    monkeypatch.setattr(code_tasks, "ACTIVE_CODE_TASK", workspace)
    final = code_tasks.finalize_code_task.invoke({})
    assert "final tests passed" in final
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=workspace.root,
        text=True,
    ).strip()


def test_code_workspace_without_git_requires_confirmation(tmp_path, monkeypatch):
    from src import code_tasks

    monkeypatch.setattr(code_tasks.shutil, "which", lambda executable: None)
    declined = CodeTaskWorkspace.create(
        "Create a small script",
        str(tmp_path / "declined"),
        confirm_without_git=lambda reason: False,
    )
    assert declined is None
    assert list((tmp_path / "declined").iterdir()) == []

    approved = CodeTaskWorkspace.create(
        "Create a small script",
        str(tmp_path / "approved"),
        confirm_without_git=lambda reason: True,
    )
    assert approved is not None
    assert approved.git_enabled is False
    assert not (approved.root / ".git").exists()


def test_git_initialization_failure_requires_confirmation_and_removes_partial_repo(
    tmp_path,
    monkeypatch,
):
    from src import code_tasks

    real_run_git = code_tasks._run_git

    def partial_init(root, *args, **kwargs):
        if args[:1] == ("init",):
            (root / ".git").mkdir()
            return subprocess.CompletedProcess(["git", *args], 1, "", "init failed")
        return real_run_git(root, *args, **kwargs)

    monkeypatch.setattr(code_tasks.shutil, "which", lambda executable: "/usr/bin/git")
    monkeypatch.setattr(code_tasks, "_run_git", partial_init)
    workspace = CodeTaskWorkspace.create(
        "Create another script",
        str(tmp_path / "projects"),
        confirm_without_git=lambda reason: True,
    )
    assert workspace is not None
    assert not workspace.git_enabled
    assert not (workspace.root / ".git").exists()


def test_existing_code_workspace_copy_excludes_git_secrets_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "Private Agent Test")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "user.email")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "private-agent-test@example.invalid")
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("[remote]\n")
    (source / ".env").write_text("TOKEN=secret\n")
    (source / ".env.example").write_text("TOKEN=replace-me\n")
    (source / ".gitignore").write_text("ignored.txt\n")
    (source / "ignored.txt").write_text("ignored by source project\n")
    (source / "private.pem").write_text("private key\n")
    outside_secret = tmp_path / "outside-secret"
    outside_secret.write_text("external secret\n")
    (source / "linked_secret").symlink_to(outside_secret)
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "cache.pyc").write_bytes(b"cache")
    (source / "app.py").write_text("print('unchanged')\n")

    workspace = CodeTaskWorkspace.create(
        "Modify this Python project",
        str(tmp_path / "projects"),
        source_path=str(source),
    )
    assert workspace is not None
    assert workspace.git_enabled
    assert (workspace.root / "app.py").read_text() == "print('unchanged')\n"
    assert not (workspace.root / ".git" / "config").read_text().count("[remote]")
    assert not subprocess.check_output(
        ["git", "remote"],
        cwd=workspace.root,
        text=True,
    ).strip()
    assert not (workspace.root / ".env").exists()
    assert (workspace.root / ".env.example").exists()
    assert not (workspace.root / "private.pem").exists()
    assert not (workspace.root / "linked_secret").exists()
    assert not (workspace.root / "__pycache__").exists()
    assert (source / ".env").exists()
    assert (workspace.root / "ignored.txt").exists()
    tracked_files = subprocess.check_output(
        ["git", "ls-files"],
        cwd=workspace.root,
        text=True,
    ).splitlines()
    assert "ignored.txt" not in tracked_files


def test_code_task_blocks_manual_git_shell_commands(monkeypatch):
    from src import code_tasks

    monkeypatch.setattr(code_tasks, "ACTIVE_CODE_TASK", object())
    with patch("src.tools.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "src.tools.console.input", return_value="y"
    ):
        result = run_shell_command.invoke({"command": "git status"})
    assert "manual Git commands are disabled" in result


def test_persistent_memory(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "test_session_1"

    memory.save_message(session_id, "human", "Hello agent")
    memory.save_message(session_id, "ai", "Hello human")

    history = memory.load_history(session_id)
    assert len(history) == 2
    assert history[0].content == "Hello agent"
    assert history[1].content == "Hello human"

    memory.save_summary(session_id, "Test summary of session.")
    summaries = memory.get_all_episodic_summaries()
    assert len(summaries) == 1
    assert summaries[0] == "Test summary of session."
    assert len(memory.load_history(session_id)) == 2
    assert memory.get_latest_session_id() == session_id
    assert memory.list_sessions() == [session_id]


def test_memory_summary_retrieval_is_bounded_and_chronological(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    for index in range(8):
        memory.save_summary(f"session-{index}", f"summary-{index}")
    assert memory.get_all_episodic_summaries(limit=3) == [
        "summary-5",
        "summary-6",
        "summary-7",
    ]
    memory.close()
    memory.close()


def test_memory_summaries_are_session_scoped(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("session-a", "private summary")
    memory.save_summary("session-b", "other session")
    assert memory.get_all_episodic_summaries(session_id="session-a") == [
        "private summary"
    ]
    memory.close()


def test_memory_retention_prunes_messages_and_summaries(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("old", "human", "old")
    memory.save_summary("old", "old summary")
    memory.save_message("new", "human", "new")
    with memory.conn:
        memory.conn.execute(
            "UPDATE chat_history SET timestamp = '2000-01-01 00:00:00' "
            "WHERE session_id = 'old'"
        )
    assert memory.prune_history(30) == 2
    assert memory.load_history("old") == []
    assert memory.get_all_episodic_summaries(session_id="old") == []
    assert [message.content for message in memory.load_history("new")] == ["new"]
    memory.close()


def test_memory_summaries_are_bounded(temp_db, monkeypatch):
    monkeypatch.setattr("src.database.MAX_SUMMARY_CHARS", 12)
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("session", "x" * 100)
    assert memory.get_all_episodic_summaries(session_id="session") == ["x" * 12]
    memory.close()


def test_context_budget_preserves_newest_messages():
    history = [
        HumanMessage(content="older " * 200),
        AIMessage(content="middle " * 80),
        HumanMessage(content="newest"),
    ]
    bounded = trim_history_to_context_budget(history, "question", 30)
    assert bounded == [history[-1]]


def test_context_budget_keeps_query_and_bounds_local_context():
    bounded = _build_bounded_user_input(
        "Local context",
        "past summary " * 10000,
        "source notes " * 10000,
        "answer this exact question",
        120,
    )
    assert "answer this exact question" in bounded
    assert _token_count(bounded) <= 120


def test_retrieved_source_citations_are_independent_of_model_response():
    documents = [
        types.SimpleNamespace(
            metadata={"source": "/docs/guide.pdf", "page": 5, "chunk": 2}
        ),
        types.SimpleNamespace(
            metadata={"source": "/docs/guide.pdf", "page": 5, "chunk": 2}
        ),
        types.SimpleNamespace(metadata={"source": "/docs/readme.md", "chunk": 0}),
    ]
    assert format_retrieved_citations(documents) == [
        "/docs/guide.pdf (page 5) (chunk 2)",
        "/docs/readme.md (chunk 0)",
    ]


def test_internet_probe_uses_per_call_timeout_without_global_socket_mutation():
    with patch("src.tools.socket.create_connection") as connect:
        assert check_internet_connection("example.test", 443, 0.25) is True
        connect.assert_called_once()
        assert connect.call_args.kwargs["timeout"] == 0.25


def test_outbound_url_rejects_local_and_credentialed_destinations():
    for url in (
        "http://127.0.0.1/admin",
        "http://localhost/private",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
    ):
        with pytest.raises(ValueError):
            validate_outbound_url(url)


def test_outbound_url_rejects_domains_resolving_to_private_addresses(monkeypatch):
    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("10.0.0.12", 80)),
        ],
    )
    with pytest.raises(ValueError, match="non-public"):
        validate_outbound_url("http://internal.example.test/")


@pytest.mark.asyncio
async def test_connect_backend_pins_validated_public_address(monkeypatch):
    backend = _PublicOnlyNetworkBackend()
    connect = AsyncMock(return_value="connected")
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("1.1.1.1", 443)),
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ],
    )
    result = await backend.connect_tcp("public.example.test", 443)
    assert result == "connected"
    assert connect.await_args.args[0] == "1.1.1.1"


@pytest.mark.asyncio
async def test_connect_backend_falls_back_from_unavailable_localhost_ipv6(monkeypatch):
    backend = _PublicOnlyNetworkBackend(allow_loopback=True)
    attempts = []

    async def connect(address, port, **kwargs):
        attempts.append(address)
        if address == "::1":
            import httpcore
            raise httpcore.ConnectError("IPv6 listener unavailable")
        return "connected"

    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (10, 1, 6, "", ("::1", 11434, 0, 0)),
            (2, 1, 6, "", ("127.0.0.1", 11434)),
        ],
    )
    assert await backend.connect_tcp("localhost", 11434) == "connected"
    assert attempts == ["::1", "127.0.0.1"]


def test_sync_connect_backend_falls_back_to_second_validated_address(monkeypatch):
    from src.tools import _PublicOnlySyncNetworkBackend
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
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (10, 1, 6, "", ("::1", 11434, 0, 0)),
            (2, 1, 6, "", ("127.0.0.1", 11434)),
        ],
    )
    assert backend.connect_tcp("localhost", 11434) == "connected"
    assert attempts == ["::1", "127.0.0.1"]


@pytest.mark.asyncio
async def test_connect_backend_rejects_mixed_public_private_dns(monkeypatch):
    backend = _PublicOnlyNetworkBackend()
    connect = AsyncMock()
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("1.1.1.1", 443)),
            (2, 1, 6, "", ("10.0.0.5", 443)),
        ],
    )
    with pytest.raises(OSError, match="non-public"):
        await backend.connect_tcp("mixed.example.test", 443)
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_private_provider_transport_allows_only_explicit_private_literals(monkeypatch):
    backend = _PublicOnlyNetworkBackend(allow_loopback=True)
    connect = AsyncMock(return_value="connected")
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda host, port, *args: [(2, 1, 6, "", (host, port))],
    )
    assert await backend.connect_tcp("192.168.1.20", 8080) == "connected"
    connect.assert_awaited_once()

    for hostname, resolved in (
        ("provider.example.test", "192.168.1.20"),
        ("169.254.169.254", "169.254.169.254"),
    ):
        monkeypatch.setattr(
            "src.tools.socket.getaddrinfo",
            lambda *args, address=resolved, **kwargs: [
                (2, 1, 6, "", (address, 443))
            ],
        )
        with pytest.raises(OSError, match="non-public"):
            await backend.connect_tcp(hostname, 443)


@pytest.mark.asyncio
async def test_http_redirect_to_local_network_is_rejected(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"Location": "http://127.0.0.1/admin"},
        )
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(ValueError, match="non-public"):
            await _get_limited_response(client, "http://public.example.test/", 100)


@pytest.mark.asyncio
async def test_http_response_size_limit_is_enforced(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "src.tools.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"too large")
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(ValueError, match="size limit"):
            await _get_limited_response(client, "http://public.example.test/", 4)


@pytest.mark.asyncio
async def test_http_total_timeout_releases_concurrency_slot(monkeypatch):
    import src.tools as tools

    class SlowClient:
        def stream(self, *args, **kwargs):
            class SlowResponse:
                async def __aenter__(self):
                    await asyncio.sleep(0.1)

                async def __aexit__(self, *args):
                    pass
            return SlowResponse()

    monkeypatch.setattr(tools, "NETWORK_REQUEST_TIMEOUT", 0.01)
    monkeypatch.setattr(tools, "validate_outbound_url", lambda url: url)
    with pytest.raises(asyncio.TimeoutError):
        await _get_limited_response(SlowClient(), "https://public.test/", 100)
    assert tools._network_slots.acquire(blocking=False)
    tools._network_slots.release()


def test_failed_download_preserves_existing_destination(temp_workspace, monkeypatch):
    import src.tools as tools

    destination = temp_workspace / "existing.txt"
    destination.write_text("original", encoding="utf-8")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(tools, "has_internet_connection", lambda: True)
    monkeypatch.setattr(tools, "_public_only_async_client", lambda **kwargs: FakeClient())

    async def fail_fetch(*args, **kwargs):
        raise OSError("network dropped")

    monkeypatch.setattr(tools, "_get_limited_response", fail_fetch)
    result = download_web_file.invoke({
        "url": "https://public.example.test/file.txt",
        "save_path": "existing.txt",
    })
    assert "network dropped" in result
    assert destination.read_text(encoding="utf-8") == "original"
    assert list(temp_workspace.glob(".existing.txt.*.part")) == []


@pytest.mark.asyncio
async def test_offline_research_can_continue_with_incomplete_data():
    from src import agent

    with patch("src.agent.has_internet_connection", return_value=False), \
         patch("src.agent.sys.stdin", MagicMock(isatty=lambda: True)), \
         patch.object(agent.console, "input", return_value="c"):
        allowed, message = await authorize_network_research(
            "web_search", {"query": "example"}
        )
    assert allowed is False
    assert "incomplete" in message


def test_rag_chunker_splits_long_documents_with_overlap():
    chunks = _split_into_chunks("x" * 2500, chunk_size=1200, overlap=200)
    assert len(chunks) == 3
    assert all(chunks)


def test_online_api_url_normalizes_v1_and_requires_secure_remote_transport():
    assert _validate_online_base_url("https://api.example.test") == (
        "https://api.example.test/v1"
    )
    assert _validate_online_base_url("http://localhost:8080/v1/") == (
        "http://localhost:8080/v1"
    )
    with pytest.raises(ValueError):
        _validate_online_base_url("http://api.example.test/v1")
    with pytest.raises(ValueError):
        _validate_online_base_url("https://user:secret@api.example.test/v1")


def test_online_model_uses_openai_compatible_client_with_session_key():
    adapter = types.ModuleType("langchain_openai")
    adapter.ChatOpenAI = MagicMock(return_value=object())
    with patch.dict(sys.modules, {"langchain_openai": adapter}):
        model = _make_online_chat_model(
            "https://api.example.test/v1", "test-model", "session-secret"
        )
    assert model is not None
    adapter.ChatOpenAI.assert_called_once()
    assert adapter.ChatOpenAI.call_args.kwargs["base_url"] == "https://api.example.test/v1"
    assert adapter.ChatOpenAI.call_args.kwargs["api_key"] == "session-secret"
    assert adapter.ChatOpenAI.call_args.kwargs["http_client"] is not None
    assert adapter.ChatOpenAI.call_args.kwargs["http_async_client"] is not None
    asyncio.run(close_outbound_http_clients())


def test_online_model_listing_uses_safe_transport_and_session_key(monkeypatch):
    import src.agent as agent

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "model-a"}]}

    class FakeHttpClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            self.url = url
            return FakeResponse()

    client = FakeHttpClient()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(agent, "public_only_sync_client", factory)
    monkeypatch.setattr(agent, "check_internet_connection", lambda *args: True)
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent, "APP_CONFIG", {"online_base_url": "https://api.example.test/v1"})
    monkeypatch.setattr(agent, "getpass", lambda prompt: "ephemeral-test-key")
    monkeypatch.setattr(agent, "_make_online_chat_model", lambda *args: "online-model")
    monkeypatch.setattr(
        agent.console,
        "input",
        MagicMock(side_effect=["", "y", "", "n", "n"]),
    )
    selected = select_online_model()
    assert selected["model"] == "online-model"
    assert selected["model_name"] == "model-a"
    assert "api_key" not in selected
    assert factory.call_args.kwargs["allow_loopback"] is True
    assert client.url == "https://api.example.test/v1/models"
    assert "ephemeral-test-key" not in repr(selected)


def test_online_model_authentication_failure_is_redacted(monkeypatch):
    import httpx
    import src.agent as agent

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "bad key secret-should-not-leak",
                request=httpx.Request("GET", "https://api.example.test/v1/models"),
                response=httpx.Response(401),
            )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(agent, "public_only_sync_client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(agent, "check_internet_connection", lambda *args: True)
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent, "APP_CONFIG", {"online_base_url": "https://api.example.test/v1"})
    monkeypatch.setattr(agent, "getpass", lambda prompt: "secret-should-not-leak")
    prompt = MagicMock(side_effect=["", "y"])
    monkeypatch.setattr(agent.console, "input", prompt)
    output = []
    monkeypatch.setattr(agent.console, "print", lambda *args, **kwargs: output.append(str(args)))

    assert select_online_model() is None
    assert all("secret-should-not-leak" not in line for line in output)
    assert prompt.call_count == 2


@pytest.mark.asyncio
async def test_network_tool_call_requires_runtime_authorization():
    with patch("src.agent.ENABLE_WEB_RESEARCH", True), \
         patch("src.agent.WEB_RESEARCH_CONSENT", "ask"), \
         patch("src.agent.AVAILABLE_TOOLS", {"web_search": MagicMock()}) as tools:
        result = await execute_tool_call(
            {"name": "web_search", "args": {"query": "private query"}, "id": "call-1"}
        )
    assert "not authorized" in result.content
    tools["web_search"].invoke.assert_not_called()


@pytest.mark.asyncio
async def test_tool_result_and_failure_are_returned_to_agent(monkeypatch):
    import src.agent as agent

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
async def test_mcp_approval_redacts_sensitive_arguments(monkeypatch):
    import src.agent as agent

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


def test_database_concurrency(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "concurrent_session"

    def write_task(idx):
        memory.save_message(session_id, "human", f"Message {idx}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(write_task, range(10)))

    history = memory.load_history(session_id)
    assert len(history) == 10

def test_sqlite_concurrent_read_write_locking(temp_db):
    """Test concurrent SQLite read and write operations under thread-safe locking."""
    memory = PersistentMemory(db_path=temp_db)
    session_id = "lock_test_session"

    def worker(idx):
        if idx % 2 == 0:
            memory.save_message(session_id, "human", f"Message {idx}")
        else:
            read_chat_history_from_sqlite.invoke({"limit": 5})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(20)))

    history = memory.load_history(session_id)
    assert len(history) > 0

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
    with patch("src.tools.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "src.tools.console.input", return_value="n"
    ), patch("subprocess.run") as run:
        result = run_shell_command.invoke({"command": "echo no"})
    assert "was not approved" in result
    run.assert_not_called()


def test_tool_catalog_shows_origin_permissions_and_effects(monkeypatch):
    import src.tools as tools

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


def test_tool_catalog_prints_each_entry_on_one_line():
    import io
    from rich.console import Console as RichConsole
    from src.agent import _print_tool_catalog

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
    with patch("src.agent.console", console):
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
    import src.tools as tools

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


def test_explicit_mcp_http_endpoints_are_peer_validated(monkeypatch):
    import src.tools as tools

    checked = []
    monkeypatch.setattr(
        tools,
        "_resolve_validated_peer",
        lambda host, port, allow_loopback: checked.append(
            (host, port, allow_loopback)
        ) or [__import__("ipaddress").ip_address("203.0.113.5")],
    )
    assert _validate_explicit_service_url(
        "http://mcp.example.test:8080/sse", {"http", "https"}
    ) == "http://mcp.example.test:8080/sse"
    assert checked == [("mcp.example.test", 8080, True)]
    with pytest.raises(ValueError, match="explicit IP address"):
        _validate_explicit_service_url("wss://mcp.example.test/ws", {"ws", "wss"})


def test_mcp_http_transport_uses_pinned_client_factory(monkeypatch):
    import src.tools as tools

    captured = {}

    class FakeMCPClient:
        def __init__(self, configs):
            captured["configs"] = configs

        async def get_tools(self, server_name):
            return []

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FakeMCPClient
    monkeypatch.setattr(
        tools,
        "_resolve_validated_peer",
        lambda host, port, allow_loopback: [
            __import__("ipaddress").ip_address("203.0.113.5")
        ],
    )
    client = None
    with patch.dict(
        sys.modules,
        {
            "langchain_mcp_adapters": parent,
            "langchain_mcp_adapters.client": adapter,
        },
    ):
        assert asyncio.run(
            load_configured_mcp_tools(
                "remote",
                {
                    "transport": "streamable_http",
                    "url": "https://mcp.example.test/mcp",
                },
            )
        ) == []
        factory = captured["configs"]["remote"]["httpx_client_factory"]
        client = factory(headers={}, timeout=None, auth=None)
    try:
        assert client.trust_env is False
        assert client.follow_redirects is False
        assert client._transport._pool._network_backend._allow_loopback is True
    finally:
        asyncio.run(client.aclose())


def test_explicit_endpoint_blocks_private_hostname_resolution(monkeypatch):
    import socket
    import src.tools as tools

    monkeypatch.setattr(
        tools.socket,
        "getaddrinfo",
        lambda host, port, *_args: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", port))
        ],
    )
    with pytest.raises(OSError, match="non-public address"):
        tools._resolve_validated_peer(
            "mcp.example.test", 8080, allow_loopback=True
        )


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
        "src.code_tasks._isolated_command",
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


def test_duckduckgo_search_uses_bounded_pinned_http_client(monkeypatch):
    import src.tools as tools

    html = b"""
    <div class="result">
      <a class="result__a" href="https://docs.example.test">Docs</a>
      <a class="result__snippet">A useful result</a>
    </div>
    """

    class FakeResponse:
        is_redirect = False
        headers = {"content-length": str(len(html))}
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield html

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream(self, method, url, params=None):
            assert method == "GET"
            assert params == {"q": "private agent"}
            return FakeResponse()

    monkeypatch.setattr(tools, "validate_outbound_url", lambda url: url)
    result = _search_duckduckgo(
        "private agent",
        client_factory=lambda **kwargs: FakeClient(),
    )
    assert "Title: Docs" in result
    assert "Snippet: A useful result" in result
    assert "https://docs.example.test" in result


def test_download_restricted_extension(temp_workspace):
    res = download_web_file.invoke({"url": "http://example.com/script.sh", "save_path": "script.sh"})
    assert "Error" in res
    assert "restricted by security policy" in res

def test_sqlite_history_tools(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session_a", "human", "Test message for sqlite tools")

    read_res = read_chat_history_from_sqlite.invoke({"limit": 5})
    assert "Test message for sqlite tools" in read_res

    del_res = delete_chat_history_from_sqlite.invoke({"session_id": "session_a"})
    assert "Success" in del_res

    wipe_res = delete_chat_history_from_sqlite.invoke({})
    assert "Success" in wipe_res

def test_shell_command_tool(temp_workspace):
    with patch("src.tools.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "src.tools.console.input", return_value="y"
    ):
        res = run_shell_command.invoke({"command": "echo 'sandbox integration test'"})
    assert "Exit Code: 0" in res
    assert "sandbox integration test" in res

def test_markdown_skill_loading(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    
    skill_file = skill_dir / "python_expert.md"
    skill_file.write_text("# Python Expert\nWrite secure and robust python code.", encoding="utf-8")

    empty_file = skill_dir / "empty_skill.md"
    empty_file.write_text("", encoding="utf-8")

    skills = load_skills_from_folder(str(skill_dir))
    assert "python_expert" in skills
    assert "empty_skill" in skills

    matched = match_skill_by_relevancy("I need help with python coding", skills)
    assert matched is not None
    assert matched.name == "Python Expert"


def test_create_skill_tool_writes_markdown_and_updates_active_registry(tmp_path):
    registry = {}
    set_active_skill_runtime(str(tmp_path / "skills"), registry)
    result = create_skill.invoke({
        "name": "Python Testing",
        "description": "Help design focused Python tests.",
        "instructions": "Prefer pytest and cover important edge cases.",
    })
    skill_file = tmp_path / "skills" / "python_testing.md"
    assert "Created skill 'Python Testing'" in result
    assert skill_file.is_file()
    assert "# Python Testing" in skill_file.read_text(encoding="utf-8")
    assert "python_testing" in registry
    assert registry["python_testing"].name == "Python Testing"
    assert match_skill_by_relevancy(
        "I need help writing Python tests", registry
    ) is registry["python_testing"]


def test_create_skill_tool_rejects_path_traversal_and_existing_files(tmp_path):
    skills_dir = tmp_path / "skills"
    set_active_skill_runtime(str(skills_dir), {})
    args = {
        "name": "../outside",
        "description": "Unsafe path test.",
        "instructions": "Must not create outside the configured directory.",
    }
    assert "Error creating skill" in create_skill.invoke(args)
    assert not (tmp_path / "outside.md").exists()

    args["name"] = "safe_skill"
    assert "Created skill" in create_skill.invoke(args)
    original_content = (skills_dir / "safe_skill.md").read_text(encoding="utf-8")
    assert "already exists" in create_skill.invoke(args)
    assert (skills_dir / "safe_skill.md").read_text(encoding="utf-8") == original_content


@pytest.mark.asyncio
async def test_skill_creation_requires_interactive_approval(monkeypatch, tmp_path):
    import src.agent as agent
    import src.tools as tools

    registry = {}
    set_active_skill_runtime(str(tmp_path / "skills"), registry)
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", lambda prompt: "n")
    args = {
        "name": "requested_skill",
        "description": "A user-requested skill.",
        "instructions": "Apply the requested reusable guidance.",
    }
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": args,
        "id": "create-declined",
    })
    assert "declined" in result.content
    assert not (tmp_path / "skills" / "requested_skill.md").exists()

    monkeypatch.setattr(agent.console, "input", lambda prompt: "y")
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": args,
        "id": "create-approved",
    })
    assert "Created skill 'requested_skill'" in result.content
    assert "requested_skill" in registry
    skill_catalog = next(
        entry for entry in tools.describe_tool_catalog()
        if entry["name"] == "create_skill"
    )
    assert "interactive approval" in skill_catalog["permission"]


@pytest.mark.asyncio
async def test_skill_creation_is_blocked_without_interactive_terminal(monkeypatch, tmp_path):
    import src.agent as agent

    set_active_skill_runtime(str(tmp_path / "skills"), {})
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: False))
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": {
            "name": "noninteractive",
            "description": "Not approved.",
            "instructions": "Must not be written.",
        },
        "id": "create-noninteractive",
    })
    assert "requires interactive user approval" in result.content
    assert not (tmp_path / "skills" / "noninteractive.md").exists()


def test_webcam_capture_keeps_image_transient_and_releases_device(monkeypatch):
    class CapturedFrame:
        shape = (480, 640, 3)

    class EncodedImage:
        def tobytes(self):
            return b"fake-jpeg-frame"

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, CapturedFrame()

        def release(self):
            self.released = True

    capture = FakeCapture()
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = lambda index: capture
    cv2.imencode = lambda *args: (True, EncodedImage())
    cv2.IMWRITE_JPEG_QUALITY = 1
    with patch.dict(sys.modules, {"cv2": cv2}):
        result = capture_webcam_image.invoke({"device_index": 0})
        match = __import__("re").search(r"camera-image:([0-9a-f]{32})", result)
        assert match
        message = captured_image_message(f"camera-image:{match.group(1)}")
    assert capture.released
    assert message.content[0]["type"] == "text"
    assert message.content[1]["type"] == "image_url"
    assert "ZmFrZS1qcGVnLWZyYW1l" in message.content[1]["image_url"]["url"]
    from langchain_ollama import ChatOllama
    converted = ChatOllama(model="test")._convert_messages_to_ollama_messages(
        [message]
    )
    assert converted[0]["role"] == "user"
    assert converted[0]["images"] == ["ZmFrZS1qcGVnLWZyYW1l"]
    assert consume_captured_image(f"camera-image:{match.group(1)}") is None


def test_microphone_transcription_uses_local_model_without_downloading(
    monkeypatch, tmp_path
):
    class FakeSoundDevice:
        def rec(self, frame_count, **kwargs):
            assert frame_count == 32000
            assert kwargs["device"] == 2
            return types.SimpleNamespace(tobytes=lambda: b"\0\0" * frame_count)

        def wait(self):
            return None

    class FakeWhisperModel:
        def __init__(self, model_path, **kwargs):
            assert model_path == str(tmp_path)
            assert kwargs["local_files_only"] is True
            assert kwargs["device"] == "cpu"

        def transcribe(self, audio_file):
            assert audio_file.read(4) == b"RIFF"
            return [types.SimpleNamespace(text=" local words ")], object()

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.rec = FakeSoundDevice().rec
    sounddevice.wait = FakeSoundDevice().wait
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setenv("PRIVATE_AGENT_WHISPER_MODEL_PATH", str(tmp_path))
    with patch.dict(
        sys.modules,
        {"sounddevice": sounddevice, "faster_whisper": faster_whisper},
    ):
        result = record_microphone_audio.invoke({
            "duration_seconds": 2,
            "device_index": 2,
        })
    assert "local words" in result
    assert "Offline microphone transcript" in result


def test_microphone_device_listing_reports_available_input_indexes():
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.query_devices = lambda: [
        {"name": "Output only", "max_input_channels": 0},
        {"name": "Internal microphone", "max_input_channels": 1},
    ]
    with patch.dict(sys.modules, {"sounddevice": sounddevice}):
        result = list_microphone_devices.invoke({})
    assert "1: Internal microphone (input channels: 1)" in result
    assert "Output only" not in result


@pytest.mark.asyncio
async def test_media_tools_require_local_runtime_authorization(monkeypatch):
    import src.agent as agent

    camera = MagicMock()
    camera.invoke.return_value = "should not run"
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"capture_webcam_image": camera})
    result = await agent.execute_tool_call({
        "name": "capture_webcam_image",
        "args": {"device_index": 0},
        "id": "camera-no-consent",
    })
    assert "local Ollama session" in result.content
    camera.invoke.assert_not_called()

    result = await agent.execute_tool_call(
        {
            "name": "capture_webcam_image",
            "args": {"device_index": 0},
            "id": "camera-no-vision",
        },
        local_media_allowed=True,
        media_capture_authorized=True,
        vision_supported=False,
    )
    assert "does not declare vision support" in result.content
    camera.invoke.assert_not_called()

def test_malformed_config_handling(tmp_path, monkeypatch):
    conf_path = tmp_path / ".private_agent.conf"
    conf_path.write_text("{ malformed_json: ", encoding="utf-8")
    monkeypatch.setattr("src.config.CONFIG_FILE_PATH", conf_path)

    config = load_or_create_config()
    assert isinstance(config, dict)
    assert "default_model_temperature" in config

@patch("src.rag.OllamaEmbeddings")
@patch("src.rag.Chroma")
def test_rag_initialization_mocked(mock_chroma_class, mock_embeddings_class, tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "notes.txt").write_text("Private agent vector search test content.", encoding="utf-8")

    mock_vectorstore = MagicMock()
    mock_chroma_class.from_documents.return_value = mock_vectorstore

    vs = initialize_knowledge_base(str(doc_dir))
    assert vs is not None

def test_rag_initialization_with_invalid_or_empty_paths():
    """Ensure initialize_knowledge_base returns None immediately for empty, None, or invalid paths."""
    assert initialize_knowledge_base(None) is None
    assert initialize_knowledge_base("") is None
    assert initialize_knowledge_base("  ") is None
    assert initialize_knowledge_base("/nonexistent/directory/path/12345") is None

@patch("src.rag.OllamaEmbeddings", side_effect=Exception("Connection refused"))
def test_rag_embedding_failure_fallback(mock_embeddings_class):
    vs = initialize_knowledge_base("/dummy/path")
    assert vs is None


def test_rag_corrupt_index_state_fails_closed_before_embedding(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    state_path = index / ".index_state.json"
    state_path.write_text("{broken", encoding="utf-8")
    with patch("src.rag.OllamaEmbeddings") as embeddings:
        assert initialize_knowledge_base(str(docs), index_path=str(index)) is None
    embeddings.assert_not_called()
    assert state_path.read_text(encoding="utf-8") == "{broken"


def test_rag_guided_recovery_preserves_corrupt_index(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("rebuild this index", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / ".index_state.json").write_text("{bad", encoding="utf-8")
    (index / "chroma.sqlite3").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr("src.rag.OllamaEmbeddings", lambda **kwargs: object())
    vectorstore = MagicMock()
    monkeypatch.setattr("src.rag.Chroma", MagicMock())
    monkeypatch.setattr(
        "src.rag.Chroma.from_documents", MagicMock(return_value=vectorstore)
    )

    retriever = initialize_knowledge_base(
        str(docs),
        index_path=str(index),
        confirm_rebuild=lambda detail: "unreadable" in detail,
    )
    assert retriever is not None
    backups = list(tmp_path.glob("index.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "chroma.sqlite3").read_text(encoding="utf-8") == "preserve"
    assert _read_index_state(index / ".index_state.json")


def test_rag_limits_file_size_and_keeps_state_write_atomic(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "large.txt").write_text("this is larger than allowed", encoding="utf-8")
    index = tmp_path / "index"
    monkeypatch.setattr("src.rag.RAG_MAX_FILE_BYTES", 4)

    class EmptyVectorStore:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("src.rag.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("src.rag.Chroma", EmptyVectorStore)
    retriever = initialize_knowledge_base(str(docs), index_path=str(index))
    assert retriever is not None
    assert _read_index_state(index / ".index_state.json") == {}

    state_path = index / ".index_state.json"
    original_state = '{"unchanged": {"mtime_ns": 1, "size": 1}}'
    state_path.write_text(original_state, encoding="utf-8")
    with patch("src.rag.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            _write_index_state_atomic(state_path, {"new": {"mtime_ns": 2, "size": 2}})
    assert state_path.read_text(encoding="utf-8") == original_state
    assert list(index.glob("*.tmp")) == []
    assert list(index.glob(".*.tmp")) == []


def test_rag_vector_failure_does_not_advance_index_state(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "changed.txt"
    source.write_text("updated document content", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    state_path = index / ".index_state.json"
    old_state = {
        str(source.resolve()): {"mtime_ns": 1, "size": 1}
    }
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    class FailingVectorStore:
        def __init__(self, **kwargs):
            pass

        def get(self, where):
            return {"ids": []}

        def add_documents(self, documents, ids):
            raise RuntimeError("embedding write failed")

    monkeypatch.setattr("src.rag.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("src.rag.Chroma", FailingVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is None
    assert _read_index_state(state_path) == old_state


def test_rag_enforces_corpus_byte_limit(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.txt"
    second = docs / "b.txt"
    first.write_text("abc", encoding="utf-8")
    second.write_text("def", encoding="utf-8")
    index = tmp_path / "index"
    monkeypatch.setattr("src.rag.RAG_MAX_FILE_BYTES", 10)
    monkeypatch.setattr("src.rag.RAG_MAX_CORPUS_BYTES", 4)

    class RecordingVectorStore:
        @classmethod
        def from_documents(cls, documents, embeddings, **kwargs):
            cls.sources = {doc.metadata["source"] for doc in documents}
            return cls()

    monkeypatch.setattr("src.rag.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("src.rag.Chroma", RecordingVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is not None
    assert RecordingVectorStore.sources == {str(first.resolve())}
    assert set(_read_index_state(index / ".index_state.json")) == {
        str(first.resolve())
    }


def test_rag_enforces_pdf_page_and_document_chunk_limits(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "too-many-pages.pdf").write_bytes(b"pdf")
    (docs / "too-many-chunks.md").write_text("x" * 2400, encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    monkeypatch.setattr("src.rag.RAG_MAX_PDF_PAGES", 1)
    monkeypatch.setattr("src.rag.RAG_MAX_DOCUMENTS", 1)
    pdf_module = types.ModuleType("pypdf")
    pdf_module.PdfReader = lambda path: types.SimpleNamespace(
        pages=[object(), object()]
    )
    monkeypatch.setitem(sys.modules, "pypdf", pdf_module)

    class EmptyVectorStore:
        def __init__(self, **kwargs):
            pass

        def get(self, where):
            return {"ids": []}

    monkeypatch.setattr("src.rag.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("src.rag.Chroma", EmptyVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is not None
    assert _read_index_state(index / ".index_state.json") == {}


@pytest.mark.asyncio
async def test_agent_shutdown_closes_resources_and_restores_dynamic_tools(
    monkeypatch, tmp_path
):
    import src.agent as agent

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
    import src.agent as agent

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
    inputs = MagicMock(side_effect=["", "", "1", "", "hello", "exit"])
    monkeypatch.setattr(agent.console, "input", inputs)

    await run_agent_cli_async()

    inputs.side_effect = ["", "", "y", "1", "", "exit"]
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


@pytest.mark.skipif(
    not shutil.which("bwrap") or not shutil.which("prlimit"),
    reason="Bubblewrap and prlimit are required for isolation integration test",
)
def test_generated_project_test_command_isolated_from_host_files(tmp_path):
    import socket

    project = tmp_path / "project"
    project.mkdir()
    host_secret = tmp_path / "host-secret.txt"
    host_secret.write_text("must not be readable", encoding="utf-8")
    host_listener = socket.socket()
    host_listener.bind(("127.0.0.1", 0))
    host_listener.listen()
    host_port = host_listener.getsockname()[1]
    script = (
        "from pathlib import Path; import resource; import socket; import sys\n"
        f"secret = Path({str(host_secret)!r})\n"
        "try: secret.read_text()\nexcept OSError: pass\n"
        "else: sys.exit(9)\n"
        f"assert socket.socket().connect_ex(('127.0.0.1', {host_port})) != 0\n"
        "assert resource.getrlimit(resource.RLIMIT_CPU)[0] <= 180\n"
        "Path('allowed.txt').write_text('workspace is writable')\n"
    )
    try:
        command, environment = _isolated_command(
            [sys.executable, "-c", script], project
        )
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, env=environment
        )
        assert result.returncode == 0, result.stderr
        assert (project / "allowed.txt").read_text(encoding="utf-8") == "workspace is writable"
    finally:
        host_listener.close()


def test_project_runner_runs_all_detected_suites_after_failure(tmp_path, monkeypatch):
    import src.code_tasks as code_tasks

    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_app.py").write_text(
        "def test_example():\n    assert True\n", encoding="utf-8"
    )
    (project / "package.json").write_text(
        json.dumps({"scripts": {"test": "node test.js"}}), encoding="utf-8"
    )
    workspace = CodeTaskWorkspace(project, "project", False)
    commands = []
    outcomes = iter([
        subprocess.CompletedProcess([], 1, "", "python suite failed"),
        subprocess.CompletedProcess([], 0, "npm suite passed", ""),
    ])
    monkeypatch.setattr(code_tasks, "_isolated_command", lambda command, root: (
        commands.append(list(command)) or list(command),
        {"PATH": "/usr/bin"},
    ))
    monkeypatch.setattr(code_tasks.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    monkeypatch.setattr(code_tasks.subprocess, "run", lambda *args, **kwargs: next(outcomes))

    result = workspace.run_tests()
    assert commands == [
        [sys.executable, "-m", "pytest", "-q", "--cache-clear"],
        ["npm", "test"],
    ]
    assert "python suite failed" in result
    assert "Passed: npm test" in result
    assert not result.startswith("Tests passed:")


def test_project_runner_rejects_ignored_test_files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    (project / ".gitignore").write_text("ignored_test.py\n", encoding="utf-8")
    (project / "ignored_test.py").write_text(
        "def test_ignored():\n    assert False\n", encoding="utf-8"
    )
    workspace = CodeTaskWorkspace(project, "project", True)
    assert "No unit-test file found" in workspace.run_tests()


def test_project_runner_reports_missing_runner_and_timeout(tmp_path, monkeypatch):
    import src.code_tasks as code_tasks

    project = tmp_path / "project"
    project.mkdir()
    (project / "test_app.py").write_text(
        "def test_example():\n    assert True\n", encoding="utf-8"
    )
    workspace = CodeTaskWorkspace(project, "project", False)
    real_find_spec = code_tasks.importlib.util.find_spec
    monkeypatch.setattr(code_tasks.importlib.util, "find_spec", lambda name: None)
    assert "no supported runner is available" in workspace.run_tests()

    monkeypatch.setattr(code_tasks.importlib.util, "find_spec", real_find_spec)
    monkeypatch.setattr(code_tasks.shutil, "which", lambda binary: "/usr/bin/python")
    monkeypatch.setattr(
        code_tasks,
        "_isolated_command",
        lambda command, root: (list(command), {"PATH": "/usr/bin"}),
    )
    monkeypatch.setattr(
        code_tasks.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("pytest", 1)
        ),
    )
    assert "timed out after 180 seconds" in workspace.run_tests()


def test_autonomous_error_reflection_simulation():
    tool_error_result = "Error: File 'nonexistent.py' not found."
    if "error" in tool_error_result.lower() or "traceback" in tool_error_result.lower():
        reflection_feedback = tool_error_result + "\n[System Reflection Prompt]: Your execution encountered an error/traceback. Analyze why it failed, correct your approach, and try again."
    
    assert "[System Reflection Prompt]" in reflection_feedback
    assert "Error: File 'nonexistent.py' not found." in reflection_feedback

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

@patch("src.agent.ChatOllama")
def test_model_fallback_on_tool_incompatibility(mock_chat_ollama):
    mock_primary = MagicMock()
    mock_primary.bind_tools.side_effect = Exception("Model does not support tools")
    mock_secondary = MagicMock()

    mock_chat_ollama.side_effect = [mock_primary, mock_secondary]

    model = get_robust_chat_model("primary_model", "fallback_model", tools=[])
    assert model is not None

@patch("src.agent.ChatOllama")
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


@patch("src.agent.ChatOllama")
def test_model_fallback_does_not_hide_unrecognized_configuration_error(mock_chat_ollama):
    mock_chat_ollama.side_effect = ValueError("invalid temperature configuration")
    assert get_robust_chat_model("primary_model", "fallback_model") is None
    mock_chat_ollama.assert_called_once()


@patch("src.agent.ollama.Client")
def test_model_capabilities_are_read_from_ollama_metadata(mock_client):
    from src.agent import inspect_model_capabilities

    mock_client.return_value.show.return_value = {
        "capabilities": ["completion", "tools", "vision", "thinking"]
    }
    capabilities = inspect_model_capabilities("local-model")
    assert capabilities["tools"] is True
    assert capabilities["function_calls"] is True
    assert capabilities["vision"] is True
    assert capabilities["thinking"] is True
    assert capabilities["audio"] is False
    mock_client.assert_called_once()

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
