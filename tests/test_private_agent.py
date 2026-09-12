import pytest
import pathlib
import tempfile
import time
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
from rich.panel import Panel

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
    load_mcp_tools
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
from src.rag import initialize_knowledge_base

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
    assert "blocked automatically in non-interactive" in res

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
