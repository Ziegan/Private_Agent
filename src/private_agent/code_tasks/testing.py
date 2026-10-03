"""Discover and run project tests inside the code-task sandbox."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import (
    CODE_TASK_FAILURE_OUTPUT_CHARS,
    CODE_TASK_SUCCESS_OUTPUT_CHARS,
    CODE_TASK_TEST_TIMEOUT,
)

if TYPE_CHECKING:
    from . import CodeTaskWorkspace


def run_project_tests(
    workspace: "CodeTaskWorkspace",
    timeout: int = CODE_TASK_TEST_TIMEOUT,
) -> str:
    from . import _TEST_FILE, _is_git_ignored, _isolated_command, _project_files

    files = []
    for path in _project_files(workspace.root):
        relative_path = path.relative_to(workspace.root).as_posix()
        if (
            _TEST_FILE.search(relative_path)
            and not _is_git_ignored(workspace.root, relative_path)
        ):
            files.append(relative_path)
    if not files:
        return "No unit-test file found. Add or update a unit test before checkpointing."

    for current_root, directories, _ in os.walk(workspace.root):
        directories[:] = [
            name
            for name in directories
            if name not in {".git", "__pycache__"}
            and not (Path(current_root) / name).is_symlink()
        ]
        bytecode_cache = Path(current_root) / "__pycache__"
        if bytecode_cache.is_dir() and not bytecode_cache.is_symlink():
            shutil.rmtree(bytecode_cache)

    python_test_files = [path for path in files if path.endswith(".py")]
    if python_test_files:
        python_tests = [
            (workspace.root / path).read_text(encoding="utf-8", errors="ignore")
            for path in python_test_files
        ]
        if importlib.util.find_spec("pytest") is not None:
            commands = [[sys.executable, "-m", "pytest", "-q", "--cache-clear"]]
        elif any("unittest.TestCase" in text for text in python_tests):
            commands = [[sys.executable, "-m", "unittest", "discover", "-v"]]
        else:
            workspace.last_test_result = (
                "Tests unavailable: Python test files require pytest or "
                "unittest.TestCase, but no supported runner is available."
            )
            return workspace.last_test_result
    else:
        commands = []

    package_json = workspace.root / "package.json"
    if package_json.exists():
        try:
            package_config = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            workspace.last_test_result = f"Tests unavailable: package.json is unreadable: {exc}"
            return workspace.last_test_result
        if (
            isinstance(package_config, dict)
            and isinstance(package_config.get("scripts"), dict)
            and package_config["scripts"].get("test")
        ):
            commands.append(["npm", "test"])
        else:
            commands.append(None)
    if (workspace.root / "go.mod").exists():
        commands.append(["go", "test", "./..."])
    if (workspace.root / "Cargo.toml").exists():
        commands.append(["cargo", "test"])
    commands = list(
        dict.fromkeys(
            tuple(command) if command is not None else None
            for command in commands
        )
    )
    commands = [
        list(command) if command is not None else None for command in commands
    ]
    if not commands:
        workspace.last_test_result = "Could not identify a test runner for this project."
        return workspace.last_test_result

    failures = []
    successes = []
    for command in commands:
        if command is None:
            failures.append("npm test skipped: package.json has no test script")
            continue
        if shutil.which(command[0]) is None:
            failures.append(
                f"{' '.join(command)} unavailable: {command[0]} is not installed"
            )
            continue
        try:
            isolated_command, environment = _isolated_command(
                command, workspace.root
            )
            result = subprocess.run(
                isolated_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{' '.join(command)} timed out after {timeout} seconds")
            continue
        except (OSError, RuntimeError) as exc:
            failures.append(
                f"{' '.join(command)} could not run in isolation: {exc}"
            )
            continue
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            successes.append(
                f"Passed: {' '.join(command)}\n"
                f"{output[-CODE_TASK_SUCCESS_OUTPUT_CHARS:]}"
            )
        else:
            failures.append(
                f"{' '.join(command)} exited {result.returncode}:\n"
                f"{output[-CODE_TASK_FAILURE_OUTPUT_CHARS:]}"
            )
    if failures:
        workspace.last_test_result = "Tests failed or incomplete:\n" + "\n".join(
            successes + failures
        )
    else:
        workspace.last_test_result = "Tests passed:\n" + "\n".join(successes)
    return workspace.last_test_result
