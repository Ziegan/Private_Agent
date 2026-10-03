import pathlib
import tempfile
import time

import pytest
from rich.console import Console
from rich.panel import Panel

from private_agent.sandbox import SandboxManager
from private_agent.tools import set_active_db_path

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
    console.print(
        Panel(summary_text, title="Test Suite Summary Report", border_style="cyan")
    )


@pytest.fixture(autouse=True)
def track_test_progress(request, session_tracker):
    test_name = request.node.name
    session_tracker.current_test = test_name
    console.print(
        f"[yellow][RUNNING][/yellow] Executing test: [bold]{test_name}[/bold]..."
    )
    started = time.time()
    try:
        yield
        elapsed = time.time() - started
        session_tracker.passed += 1
        console.print(f"[green][PASSED][/green] {test_name} ({elapsed:.3f}s)\n")
    except Exception as exc:
        elapsed = time.time() - started
        session_tracker.failed += 1
        console.print(
            f"[red][FAILED][/red] {test_name} ({elapsed:.3f}s) - Error: {exc}\n"
        )
        raise


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temporary_file:
        db_path = temporary_file.name
    set_active_db_path(db_path)
    yield db_path
    path = pathlib.Path(db_path)
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as temporary_directory:
        original_root = SandboxManager.root_dir
        SandboxManager.set_root(temporary_directory)
        yield pathlib.Path(temporary_directory)
        SandboxManager.set_root(str(original_root))
