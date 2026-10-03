import pathlib
import json
import shutil
import sys
import subprocess
from unittest.mock import patch, MagicMock
from rich.console import Console
import pytest

from private_agent.tools import (
    run_shell_command,
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

from private_agent.code_tasks import (
    CodeTaskWorkspace,
    is_code_task_request,
    _isolated_command,
)

console = Console()

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

    from private_agent import code_tasks

    monkeypatch.setattr(code_tasks, "ACTIVE_CODE_TASK", workspace)
    final = code_tasks.finalize_code_task.invoke({})
    assert "final tests passed" in final
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=workspace.root,
        text=True,
    ).strip()

def test_code_workspace_without_git_requires_confirmation(tmp_path, monkeypatch):
    from private_agent import code_tasks

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
    from private_agent import code_tasks

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
    from private_agent import code_tasks

    monkeypatch.setattr(code_tasks, "ACTIVE_CODE_TASK", object())
    with patch("private_agent.tools.shell.sys.stdin", MagicMock(isatty=lambda: True)), patch(
        "private_agent.tools.shell.console.input", return_value="y"
    ):
        result = run_shell_command.invoke({"command": "git status"})
    assert "manual Git commands are disabled" in result


@pytest.mark.parametrize(
    ("system", "available_tools", "message"),
    [
        ("Darwin", {"bwrap", "prlimit"}, "currently supported only on Linux"),
        ("Linux", {"prlimit"}, "install Bubblewrap"),
        ("Linux", {"bwrap"}, "install Bubblewrap"),
    ],
)
def test_isolated_command_fails_closed_when_isolation_is_unavailable(
    monkeypatch, tmp_path, system, available_tools, message
):
    import private_agent.code_tasks as code_tasks

    monkeypatch.setattr(code_tasks.platform, "system", lambda: system)
    monkeypatch.setattr(
        code_tasks.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in available_tools else None,
    )

    with pytest.raises(RuntimeError, match=message):
        code_tasks._isolated_command(["python", "-c", "print('no fallback')"], tmp_path)


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
    host_credentials = tmp_path / "host-home" / ".private_agent" / "credentials.json"
    host_credentials.parent.mkdir(parents=True)
    host_credentials.write_text('{"api_key":"must not be readable"}', encoding="utf-8")
    host_write = tmp_path / "host-write.txt"
    host_listener = socket.socket()
    host_listener.bind(("127.0.0.1", 0))
    host_listener.listen()
    host_port = host_listener.getsockname()[1]
    script = (
        "from pathlib import Path; import os; import resource; import socket; import sys\n"
        f"for secret_path in ({str(host_secret)!r}, {str(host_credentials)!r}):\n"
        "    try: Path(secret_path).read_text()\n"
        "    except OSError: pass\n"
        "    else: sys.exit(9)\n"
        f"assert socket.socket().connect_ex(('127.0.0.1', {host_port})) != 0\n"
        "assert resource.getrlimit(resource.RLIMIT_CPU)[0] <= 180\n"
        "assert resource.getrlimit(resource.RLIMIT_AS)[0] <= 2 * 1024**3\n"
        "assert resource.getrlimit(resource.RLIMIT_NPROC)[0] <= 128\n"
        "assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] <= 512 * 1024**2\n"
        "assert os.environ['HOME'] == '/tmp'\n"
        "assert os.environ['TMPDIR'] == '/tmp'\n"
        f"try: Path({str(host_write)!r}).write_text('must remain private')\n"
        "except OSError: pass\n"
        "else: sys.exit(10)\n"
        "Path('/tmp/isolated-write.txt').write_text('temporary sandbox write')\n"
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
        assert not host_write.exists()
    finally:
        host_listener.close()

def test_project_runner_runs_all_detected_suites_after_failure(tmp_path, monkeypatch):
    import private_agent.code_tasks as code_tasks

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
    import private_agent.code_tasks as code_tasks

    project = tmp_path / "project"
    project.mkdir()
    (project / "test_app.py").write_text(
        "def test_example():\n    assert True\n", encoding="utf-8"
    )
    workspace = CodeTaskWorkspace(project, "project", False)
    from private_agent.code_tasks import testing

    real_find_spec = testing.importlib.util.find_spec
    monkeypatch.setattr(testing.importlib.util, "find_spec", lambda name: None)
    assert "no supported runner is available" in workspace.run_tests()

    monkeypatch.setattr(testing.importlib.util, "find_spec", real_find_spec)
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
