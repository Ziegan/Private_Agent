"""Git checkpoint and finalization operations for isolated code tasks."""

import re
from pathlib import Path
from typing import Callable, Sequence

from . import _TEST_FILE, _is_git_ignored, _is_likely_secret, _project_files


def commit_paths(
    workspace,
    paths: Sequence[str],
    message: str,
    *,
    require_test: bool,
    run_git: Callable,
) -> str:
    if not workspace.git_enabled:
        return "Git is disabled for this project by user confirmation; no Git checkpoint was created."
    try:
        candidates = workspace._candidate_paths()
    except (RuntimeError, OSError) as exc:
        return f"Error: {exc}"
    if not candidates:
        return "No project changes are available to checkpoint."

    requested = {str(path).replace("\\", "/") for path in paths if str(path).strip()}
    if not requested:
        return "Error: Provide the changed project paths to include in the checkpoint."
    invalid = [
        path for path in requested
        if Path(path).is_absolute()
        or ".." in Path(path).parts
        or path not in candidates
    ]
    if invalid:
        return f"Error: Paths are outside the current changed project files: {invalid}"

    if require_test:
        test_paths = {
            path.relative_to(workspace.root).as_posix()
            for path in _project_files(workspace.root)
            if _TEST_FILE.search(path.relative_to(workspace.root).as_posix())
            and not _is_git_ignored(
                workspace.root, path.relative_to(workspace.root).as_posix()
            )
        }
        if not test_paths:
            return "Error: No unit-test file exists. Add or update a test before checkpointing."
        tracked = run_git(workspace.root, "ls-files", "-z")
        tracked_test_paths = {
            path for path in tracked.stdout.split("\0")
            if path and _TEST_FILE.search(path)
        }
        if not requested.intersection(test_paths | tracked_test_paths):
            return (
                "Error: Include at least one relevant unit-test file in this "
                "checkpoint, or ensure an existing test file is already committed."
            )
        test_result = workspace._run_tests()
        if not test_result.startswith("Tests passed:"):
            return f"Error: Checkpoint blocked because tests did not pass.\n{test_result}"

    if any(_is_likely_secret(path) for path in requested):
        return "Error: Refusing to commit a likely secret file."

    add = run_git(workspace.root, "add", "--", *sorted(requested))
    if add.returncode != 0:
        return f"Error staging files: {add.stderr.strip()}"
    staged = run_git(workspace.root, "diff", "--cached", "--name-only", "-z")
    staged_paths = {path for path in staged.stdout.split("\0") if path}
    if staged.returncode != 0 or not staged_paths or not staged_paths.issubset(requested):
        return "Error: Staged paths do not match the requested checkpoint file list."
    staged_diff = run_git(workspace.root, "diff", "--cached", "--check")
    if staged_diff.returncode != 0:
        return f"Error: Staged diff check failed: {staged_diff.stderr.strip()}"
    commit = run_git(
        workspace.root,
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        message,
    )
    if commit.returncode != 0:
        return f"Error creating local checkpoint: {commit.stderr.strip()}"
    commit_id = run_git(workspace.root, "rev-parse", "HEAD")
    if commit_id.returncode != 0:
        return "Error: Commit command succeeded but its commit ID could not be verified."
    remotes = run_git(workspace.root, "remote")
    if remotes.returncode != 0 or remotes.stdout.strip():
        return "Error: Checkpoint exists, but unexpected remote configuration was detected."
    return f"Checkpoint committed locally: {commit_id.stdout.strip()}"


def checkpoint(
    workspace,
    label: str,
    paths: Sequence[str],
    *,
    run_git: Callable,
) -> str:
    safe_label = re.sub(r"[^a-zA-Z0-9 ._-]", "", label).strip()[:72]
    if not safe_label:
        return "Error: Provide a short checkpoint label."
    return commit_paths(
        workspace,
        paths,
        f"task {workspace.task_id}: {safe_label}",
        require_test=True,
        run_git=run_git,
    )


def finalize(workspace, *, run_git: Callable) -> str:
    tests = workspace._run_tests()
    if not tests.startswith("Tests passed:"):
        return f"Final verification blocked.\n{tests}"
    if not workspace.git_enabled:
        return (
            "Tests passed, but Git initialization/checkpoints were skipped "
            "after user approval."
        )
    try:
        changed_paths = workspace._candidate_paths()
    except (RuntimeError, OSError) as exc:
        return f"Final checkpoint failed: {exc}"
    if not changed_paths:
        if not workspace.initial_commit:
            return "No task changes were committed; no verified code checkpoint exists."
        head = run_git(workspace.root, "rev-parse", "HEAD")
        if head.returncode != 0 or head.stdout.strip() == workspace.initial_commit:
            return "No code-task checkpoint was created; task completion is unverified."
        return (
            f"Task checkpoints are committed locally through "
            f"{head.stdout.strip()}; final tests passed."
        )
    return commit_paths(
        workspace,
        changed_paths,
        f"task {workspace.task_id}: verified completion",
        require_test=True,
        run_git=run_git,
    )
