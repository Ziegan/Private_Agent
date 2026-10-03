"""Creation and lifecycle for isolated code-task workspaces."""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..config import CODE_TASK_TEST_TIMEOUT


@dataclass
class CodeTaskWorkspace:
    root: Path
    task_id: str
    git_enabled: bool
    original_source: Optional[Path] = None
    last_test_result: Optional[str] = None
    initial_commit: Optional[str] = None

    @classmethod
    def create(
        cls,
        task_description: str,
        output_root: str,
        *,
        source_path: Optional[str] = None,
        confirm_without_git=None,
    ) -> Optional["CodeTaskWorkspace"]:
        from . import (
            _ask_without_git,
            _copy_ignore,
            _run_git,
            _slug,
        )

        root_parent = Path(output_root).expanduser().resolve()
        root_parent.mkdir(parents=True, exist_ok=True)
        task_id = _slug(task_description)
        destination = root_parent / task_id
        suffix = 1
        while destination.exists():
            destination = root_parent / f"{task_id}-{suffix}"
            suffix += 1
        destination.mkdir()

        source = Path(source_path).expanduser().resolve() if source_path else None
        if source:
            if (
                not source.is_dir()
                or source == destination
                or destination.is_relative_to(source)
            ):
                destination.rmdir()
                raise ValueError(
                    "Existing project source must be a valid directory outside "
                    "the output directory."
                )
            try:
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                    ignore=_copy_ignore,
                    symlinks=True,
                )
            except Exception:
                shutil.rmtree(destination)
                raise
            from . import console

            console.print(
                f"[cyan]Copied project to isolated workspace: {destination} "
                "(original left unchanged; .git, symlinks, secrets, caches, and "
                "build output excluded)[/cyan]"
            )
        else:
            from . import console

            console.print(f"[cyan]Created isolated code workspace: {destination}[/cyan]")

        git_path = shutil.which("git")
        ask = confirm_without_git or _ask_without_git

        def continue_without_git() -> "CodeTaskWorkspace":
            git_metadata = destination / ".git"
            if git_metadata.is_dir():
                shutil.rmtree(git_metadata)
            elif git_metadata.exists():
                git_metadata.unlink()
            return cls(destination, task_id, False, source)

        if not git_path:
            if not ask("git executable was not found on PATH"):
                shutil.rmtree(destination)
                return None
            return continue_without_git()

        try:
            init_result = _run_git(destination, "init", "--template=/dev/null")
        except (OSError, subprocess.TimeoutExpired) as exc:
            if not ask(f"git init failed: {exc}"):
                shutil.rmtree(destination)
                return None
            return continue_without_git()
        if init_result.returncode != 0:
            reason = f"git init failed: {init_result.stderr.strip() or 'unknown error'}"
            if not ask(reason):
                shutil.rmtree(destination)
                return None
            return continue_without_git()

        root_result = _run_git(destination, "rev-parse", "--show-toplevel")
        git_root = root_result.stdout.strip()
        if root_result.returncode != 0 or git_root not in {
            str(destination),
            "/workspace",
        }:
            shutil.rmtree(destination)
            raise RuntimeError(
                "Git repository root did not match the isolated code workspace."
            )

        remotes = _run_git(destination, "remote")
        if remotes.returncode != 0 or remotes.stdout.strip():
            shutil.rmtree(destination)
            raise RuntimeError(
                "New code workspace unexpectedly has one or more Git remotes."
            )

        identity_name = _run_git(destination, "config", "--get", "user.name")
        identity_email = _run_git(destination, "config", "--get", "user.email")
        if identity_name.returncode != 0 or identity_email.returncode != 0:
            if not sys.stdin.isatty():
                if not ask("no Git commit identity is configured"):
                    shutil.rmtree(destination)
                    return None
                return continue_without_git()
            from . import console

            name = console.input("[cyan]Name for local project commits: [/cyan]").strip()
            email = console.input("[cyan]Email for local project commits: [/cyan]").strip()
            if not name or not email:
                if not ask("a local Git author name and email are required for commits"):
                    shutil.rmtree(destination)
                    return None
                return continue_without_git()
            for key, value in (("user.name", name), ("user.email", email)):
                result = _run_git(destination, "config", "--local", key, value)
                if result.returncode != 0:
                    if not ask("repository-local Git identity configuration failed"):
                        shutil.rmtree(destination)
                        return None
                    return continue_without_git()

        gitignore = destination / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                ".env\n.env.*\n!.env.example\n*.pem\n*.key\n*.p12\n*.pfx\n"
                "__pycache__/\n.pytest_cache/\n.mypy_cache/\n.ruff_cache/\n"
                ".venv/\nvenv/\nnode_modules/\ndist/\nbuild/\ntarget/\n"
                "*.sqlite\n*.db\n.local_ai_chroma_db/\n",
                encoding="utf-8",
            )

        workspace = cls(destination, task_id, True, source)
        if source:
            checkpoint = workspace._commit_paths(
                [".gitignore", *workspace._candidate_paths(include_all=True)],
                f"task {task_id}: isolated project baseline",
                require_test=False,
            )
            if not checkpoint.startswith("Checkpoint committed"):
                if not ask("initial local Git checkpoint failed"):
                    shutil.rmtree(destination)
                    return None
                from . import console

                console.print(f"[yellow]{checkpoint}[/yellow]")
                return continue_without_git()
            workspace.initial_commit = _run_git(
                destination, "rev-parse", "HEAD"
            ).stdout.strip()
        else:
            checkpoint = workspace._commit_paths(
                [".gitignore"],
                f"task {task_id}: initialize isolated workspace",
                require_test=False,
            )
            if not checkpoint.startswith("Checkpoint committed"):
                if not ask("initial local Git checkpoint failed"):
                    shutil.rmtree(destination)
                    return None
                from . import console

                console.print(f"[yellow]{checkpoint}[/yellow]")
                return continue_without_git()
            workspace.initial_commit = _run_git(
                destination, "rev-parse", "HEAD"
            ).stdout.strip()
        return workspace

    def _candidate_paths(self, *, include_all: bool = False) -> list[str]:
        from . import (
            _is_likely_secret,
            _not_ignored_by_project_git,
            _run_git,
        )

        if include_all:
            candidates = []
            for path in self.root.rglob("*"):
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or ".git" in path.relative_to(self.root).parts
                ):
                    continue
                relative_path = path.relative_to(self.root).as_posix()
                if _is_likely_secret(relative_path):
                    continue
                if _not_ignored_by_project_git(self.root, relative_path):
                    candidates.append(relative_path)
            return candidates
        result = _run_git(
            self.root,
            "status",
            "--porcelain",
            "-z",
            "--untracked-files=all",
        )
        if result.returncode != 0:
            raise RuntimeError("Could not inspect project Git status.")
        paths = []
        entries = result.stdout.split("\0")
        index = 0
        while index < len(entries):
            entry = entries[index]
            index += 1
            if not entry:
                continue
            path = entry[3:]
            if "R" in entry[:2] or "C" in entry[:2]:
                index += 1
            paths.append(path.replace(os.sep, "/"))
        return paths

    def _run_tests(self, timeout: int = CODE_TASK_TEST_TIMEOUT) -> str:
        from .testing import run_project_tests

        return run_project_tests(self, timeout)

    def run_tests(self) -> str:
        return self._run_tests()

    def _commit_paths(
        self,
        paths: Sequence[str],
        message: str,
        *,
        require_test: bool,
    ) -> str:
        from . import _run_git
        from .checkpoints import commit_paths

        return commit_paths(
            self,
            paths,
            message,
            require_test=require_test,
            run_git=_run_git,
        )

    def checkpoint(
        self,
        label: str,
        paths: Sequence[str],
    ) -> str:
        from . import _run_git
        from .checkpoints import checkpoint

        return checkpoint(self, label, paths, run_git=_run_git)

    def finalize(self) -> str:
        from . import _run_git
        from .checkpoints import finalize

        return finalize(self, run_git=_run_git)
