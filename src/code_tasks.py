import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from rich.console import Console

console = Console()

_CODE_TASK_PATTERNS = (
    re.compile(
        r"\b(write|create|generate|build|implement|modify|edit|change|fix|"
        r"refactor|scaffold|add|remove|delete)\b.{0,80}\b(code|program|script|"
        r"software|application|app|project|function|class|module|file|bug|"
        r"unit tests?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(code|program|script|software|application|app|project|function|"
        r"class|module|file|bug|unit tests?)\b.{0,50}\b(create|generate|"
        r"modify|edit|change|fix|refactor|implement|add|remove|delete)\b",
        re.IGNORECASE,
    ),
)
_TEST_FILE = re.compile(
    r"(^|/)(tests?/|test_[^/]+\.[^/]+$|[^/]+_test\.[^/]+$|"
    r"[^/]+\.(test|spec)\.[^/]+$)"
)
_SECRET_PATH = re.compile(
    r"(^|/)(\.env($|\.)|id_rsa($|\.)|id_ed25519($|\.)|credentials\.json$)"
    r"|(\.(pem|key|p12|pfx)$)",
    re.IGNORECASE,
)
_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}


def _is_likely_secret(path: str) -> bool:
    if Path(path).name.lower() in _ENV_TEMPLATES:
        return False
    return _SECRET_PATH.search(path) is not None


def is_code_task_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_TASK_PATTERNS)


def _run_git(root: Path, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    command = ["git", "-C", str(root), *args]
    try:
        isolated_command, environment = _isolated_command(command, root)
    except RuntimeError as exc:
        return subprocess.CompletedProcess(command, 126, "", str(exc))
    return subprocess.run(
        isolated_command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def _isolated_command(command: Sequence[str], root: Path) -> tuple[list[str], dict[str, str]]:
    if platform.system() != "Linux":
        raise RuntimeError(
            "Command execution is disabled: OS-level isolation is "
            "currently supported only on Linux."
        )
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise RuntimeError(
            "Command execution is disabled: install Bubblewrap (bwrap) "
            "and prlimit to provide filesystem, network, and resource isolation."
        )
    root = root.resolve()
    rewritten_command = []
    root_prefix = str(root)
    runtime_prefix = Path(sys.prefix).resolve()
    dependency_paths = []
    for path in sys.path:
        if not path:
            continue
        resolved_path = Path(path).resolve()
        if (
            resolved_path.is_dir()
            and resolved_path.is_relative_to(Path.home())
            and resolved_path not in dependency_paths
        ):
            dependency_paths.append(resolved_path)
    dependency_mounts = {
        path: Path("/opt/private-agent-dependencies") / str(index)
        for index, path in enumerate(dependency_paths)
    }
    for argument in command:
        if argument == root_prefix or argument.startswith(root_prefix + os.sep):
            argument = "/workspace" + argument[len(root_prefix):]
        elif (
            runtime_prefix != Path("/usr")
            and (
                argument == str(runtime_prefix)
                or argument.startswith(str(runtime_prefix) + os.sep)
            )
        ):
            argument = "/opt/private-agent-runtime" + argument[len(str(runtime_prefix)):]
        else:
            for dependency_path, sandbox_path in dependency_mounts.items():
                if argument == str(dependency_path) or argument.startswith(
                    str(dependency_path) + os.sep
                ):
                    argument = str(sandbox_path) + argument[len(str(dependency_path)):]
                    break
        rewritten_command.append(argument)

    args = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--dir", "/etc",
    ]
    for system_file in (
        "/etc/ld.so.cache",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/os-release",
    ):
        if os.path.isfile(system_file):
            args.extend(["--ro-bind", system_file, system_file])
    if os.path.isdir("/etc/ssl/certs"):
        args.extend(["--dir", "/etc/ssl", "--ro-bind", "/etc/ssl/certs", "/etc/ssl/certs"])
    args.extend([
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/home",
        "--tmpfs", "/root",
        "--tmpfs", "/run",
        "--tmpfs", "/mnt",
        "--tmpfs", "/media",
        "--tmpfs", "/var",
        "--tmpfs", "/opt",
    ])
    if runtime_prefix != Path("/usr"):
        args.extend([
            "--dir", "/opt/private-agent-runtime",
            "--ro-bind", str(runtime_prefix), "/opt/private-agent-runtime",
        ])
    if dependency_mounts:
        args.extend(["--dir", "/opt/private-agent-dependencies"])
        for source, destination in dependency_mounts.items():
            args.extend([
                "--dir", str(destination),
                "--ro-bind", str(source), str(destination),
            ])
    args.extend([
        "--dir", "/workspace",
        "--bind", str(root), "/workspace",
        "--chdir", "/workspace",
    ])
    args.extend([
        prlimit,
        "--cpu=180",
        "--as=2147483648",
        "--nproc=128",
        "--fsize=536870912",
        "--",
        *rewritten_command,
    ])
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    try:
        git_config_count = min(20, max(0, int(os.environ.get("GIT_CONFIG_COUNT", "0"))))
    except ValueError:
        git_config_count = 0
    git_identity = []
    for index in range(git_config_count):
        key = os.environ.get(f"GIT_CONFIG_KEY_{index}", "").lower()
        value = os.environ.get(f"GIT_CONFIG_VALUE_{index}", "")
        if key in {"user.name", "user.email"} and len(value) <= 200:
            git_identity.append((key, value))
    if git_identity:
        environment["GIT_CONFIG_COUNT"] = str(len(git_identity))
        for index, (key, value) in enumerate(git_identity):
            environment[f"GIT_CONFIG_KEY_{index}"] = key
            environment[f"GIT_CONFIG_VALUE_{index}"] = value
    if dependency_mounts:
        environment["PYTHONPATH"] = os.pathsep.join(
            str(sandbox_path) for sandbox_path in dependency_mounts.values()
        )
    return args, environment


def _ask_without_git(reason: str) -> bool:
    if not sys.stdin.isatty():
        console.print(
            f"[red]Git is unavailable for this code task ({reason}). "
            "A user must explicitly approve continuing without Git.[/red]"
        )
        return False
    console.print(f"[yellow]Git checkpointing is unavailable: {reason}[/yellow]")
    return console.input(
        "[yellow]Proceed with this isolated code task without Git initialization "
        "and checkpoint commits? [y/N]: [/yellow]"
    ).strip().lower() == "y"


def _slug(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:6]
    return "-".join(words) or "code-task"


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".cache", ".tox", ".nox", ".next",
        "coverage", "dist", "build", "target", ".idea", ".vscode",
        ".private_agent.conf", "secrets",
    }
    return {
        name for name in names
        if name in ignored
        or (Path(directory) / name).is_symlink()
        or (
            (name == ".env" or name.startswith(".env."))
            and name.lower() not in _ENV_TEMPLATES
        )
        or _is_likely_secret(name)
    }


def _project_files(root: Path):
    excluded_directories = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        "target",
    }
    for current_root, directories, filenames in os.walk(root):
        directories[:] = [
            name
            for name in directories
            if name not in excluded_directories
            and not (Path(current_root) / name).is_symlink()
        ]
        current = Path(current_root)
        for filename in filenames:
            candidate = current / filename
            if candidate.is_file() and not candidate.is_symlink():
                yield candidate


def _is_git_ignored(root: Path, relative_path: str) -> bool:
    if not (root / ".git").exists():
        return False
    result = _run_git(
        root,
        "check-ignore",
        "-q",
        "--no-index",
        "--",
        relative_path,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(
        f"Could not determine whether project test '{relative_path}' is ignored."
    )


def _not_ignored_by_project_git(root: Path, relative_path: str) -> bool:
    result = _run_git(
        root,
        "check-ignore",
        "-q",
        "--no-index",
        "--",
        relative_path,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise RuntimeError(
        f"Could not determine whether project file {relative_path!r} is ignored."
    )


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
            if not source.is_dir() or source == destination or destination.is_relative_to(source):
                destination.rmdir()
                raise ValueError("Existing project source must be a valid directory outside the output directory.")
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
            console.print(
                f"[cyan]Copied project to isolated workspace: {destination} "
                f"(original left unchanged; .git, symlinks, secrets, caches, and build output excluded)[/cyan]"
            )
        else:
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
            reason = (
                f"git init failed: {init_result.stderr.strip() or 'unknown error'}"
            )
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
            raise RuntimeError("Git repository root did not match the isolated code workspace.")

        remotes = _run_git(destination, "remote")
        if remotes.returncode != 0 or remotes.stdout.strip():
            shutil.rmtree(destination)
            raise RuntimeError("New code workspace unexpectedly has one or more Git remotes.")

        identity_name = _run_git(destination, "config", "--get", "user.name")
        identity_email = _run_git(destination, "config", "--get", "user.email")
        if identity_name.returncode != 0 or identity_email.returncode != 0:
            if not sys.stdin.isatty():
                if not ask("no Git commit identity is configured"):
                    shutil.rmtree(destination)
                    return None
                return continue_without_git()
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
                console.print(f"[yellow]{checkpoint}[/yellow]")
                return continue_without_git()
            else:
                workspace.initial_commit = _run_git(destination, "rev-parse", "HEAD").stdout.strip()
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
                console.print(f"[yellow]{checkpoint}[/yellow]")
                return continue_without_git()
            else:
                workspace.initial_commit = _run_git(destination, "rev-parse", "HEAD").stdout.strip()
        return workspace

    def _candidate_paths(self, *, include_all: bool = False) -> list[str]:
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

    def _run_tests(self, timeout: int = 180) -> str:
        files = []
        for path in _project_files(self.root):
            relative_path = path.relative_to(self.root).as_posix()
            if (
                _TEST_FILE.search(relative_path)
                and not _is_git_ignored(self.root, relative_path)
            ):
                files.append(relative_path)
        if not files:
            self.last_test_result = "No unit-test file found. Add or update a unit test before checkpointing."
            return self.last_test_result

        for current_root, directories, _ in os.walk(self.root):
            directories[:] = [
                name for name in directories
                if name not in {".git", "__pycache__"}
                and not (Path(current_root) / name).is_symlink()
            ]
            bytecode_cache = Path(current_root) / "__pycache__"
            if bytecode_cache.is_dir() and not bytecode_cache.is_symlink():
                shutil.rmtree(bytecode_cache)

        python_test_files = [path for path in files if path.endswith(".py")]
        if python_test_files:
            python_tests = [
                (self.root / path).read_text(encoding="utf-8", errors="ignore")
                for path in python_test_files
            ]
            if importlib.util.find_spec("pytest") is not None:
                commands = [
                    [sys.executable, "-m", "pytest", "-q", "--cache-clear"]
                ]
            elif any("unittest.TestCase" in text for text in python_tests):
                commands = [[sys.executable, "-m", "unittest", "discover", "-v"]]
            else:
                commands = []
                self.last_test_result = (
                    "Tests unavailable: Python test files require pytest or "
                    "unittest.TestCase, but no supported runner is available."
                )
                return self.last_test_result
        else:
            commands = []

        package_json = self.root / "package.json"
        if package_json.exists():
            try:
                package_config = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.last_test_result = f"Tests unavailable: package.json is unreadable: {exc}"
                return self.last_test_result
            if isinstance(package_config, dict) and isinstance(
                package_config.get("scripts"), dict
            ) and package_config["scripts"].get("test"):
                commands.append(["npm", "test"])
            else:
                commands.append(None)
        if (self.root / "go.mod").exists():
            commands.append(["go", "test", "./..."])
        if (self.root / "Cargo.toml").exists():
            commands.append(["cargo", "test"])
        commands = list(dict.fromkeys(
            tuple(command) if command is not None else None
            for command in commands
        ))
        commands = [
            list(command) if command is not None else None for command in commands
        ]
        if not commands:
            self.last_test_result = "Could not identify a test runner for this project."
            return self.last_test_result

        failures = []
        successes = []
        for command in commands:
            if command is None:
                failures.append("npm test skipped: package.json has no test script")
                continue
            if shutil.which(command[0]) is None:
                failures.append(f"{' '.join(command)} unavailable: {command[0]} is not installed")
                continue
            try:
                isolated_command, environment = _isolated_command(
                    command, self.root
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
                failures.append(f"{' '.join(command)} could not run in isolation: {exc}")
                continue
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                successes.append(
                    f"Passed: {' '.join(command)}\n{output[-6000:]}"
                )
            else:
                failures.append(
                    f"{' '.join(command)} exited {result.returncode}:\n{output[-3000:]}"
                )
        if failures:
            self.last_test_result = "Tests failed or incomplete:\n" + "\n".join(
                successes + failures
            )
        else:
            self.last_test_result = "Tests passed:\n" + "\n".join(successes)
        return self.last_test_result

    def run_tests(self) -> str:
        return self._run_tests()

    def _commit_paths(
        self,
        paths: Sequence[str],
        message: str,
        *,
        require_test: bool,
    ) -> str:
        if not self.git_enabled:
            return "Git is disabled for this project by user confirmation; no Git checkpoint was created."
        try:
            candidates = self._candidate_paths()
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
                path.relative_to(self.root).as_posix()
                for path in _project_files(self.root)
                if _TEST_FILE.search(path.relative_to(self.root).as_posix())
                and not _is_git_ignored(
                    self.root, path.relative_to(self.root).as_posix()
                )
            }
            if not test_paths:
                return "Error: No unit-test file exists. Add or update a test before checkpointing."
            tracked = _run_git(self.root, "ls-files", "-z")
            tracked_test_paths = {
                path for path in tracked.stdout.split("\0")
                if path and _TEST_FILE.search(path)
            }
            if not requested.intersection(test_paths | tracked_test_paths):
                return (
                    "Error: Include at least one relevant unit-test file in this "
                    "checkpoint, or ensure an existing test file is already committed."
                )
            test_result = self._run_tests()
            if not test_result.startswith("Tests passed:"):
                return f"Error: Checkpoint blocked because tests did not pass.\n{test_result}"

        if any(_is_likely_secret(path) for path in requested):
            return "Error: Refusing to commit a likely secret file."

        add = _run_git(self.root, "add", "--", *sorted(requested))
        if add.returncode != 0:
            return f"Error staging files: {add.stderr.strip()}"
        staged = _run_git(self.root, "diff", "--cached", "--name-only", "-z")
        staged_paths = {path for path in staged.stdout.split("\0") if path}
        if staged.returncode != 0 or not staged_paths or not staged_paths.issubset(requested):
            return "Error: Staged paths do not match the requested checkpoint file list."
        staged_diff = _run_git(self.root, "diff", "--cached", "--check")
        if staged_diff.returncode != 0:
            return f"Error: Staged diff check failed: {staged_diff.stderr.strip()}"
        commit = _run_git(
            self.root,
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-m",
            message,
        )
        if commit.returncode != 0:
            return f"Error creating local checkpoint: {commit.stderr.strip()}"
        commit_id = _run_git(self.root, "rev-parse", "HEAD")
        if commit_id.returncode != 0:
            return "Error: Commit command succeeded but its commit ID could not be verified."
        remotes = _run_git(self.root, "remote")
        if remotes.returncode != 0 or remotes.stdout.strip():
            return "Error: Checkpoint exists, but unexpected remote configuration was detected."
        return f"Checkpoint committed locally: {commit_id.stdout.strip()}"

    def checkpoint(
        self,
        label: str,
        paths: Sequence[str],
    ) -> str:
        safe_label = re.sub(r"[^a-zA-Z0-9 ._-]", "", label).strip()[:72]
        if not safe_label:
            return "Error: Provide a short checkpoint label."
        return self._commit_paths(
            paths,
            f"task {self.task_id}: {safe_label}",
            require_test=True,
        )

    def finalize(self) -> str:
        tests = self._run_tests()
        if not tests.startswith("Tests passed:"):
            return f"Final verification blocked.\n{tests}"
        if not self.git_enabled:
            return (
                "Tests passed, but Git initialization/checkpoints were skipped "
                "after user approval."
            )
        try:
            changed_paths = self._candidate_paths()
        except (RuntimeError, OSError) as exc:
            return f"Final checkpoint failed: {exc}"
        if not changed_paths:
            if not self.initial_commit:
                return "No task changes were committed; no verified code checkpoint exists."
            head = _run_git(self.root, "rev-parse", "HEAD")
            if head.returncode != 0 or head.stdout.strip() == self.initial_commit:
                return "No code-task checkpoint was created; task completion is unverified."
            return (
                f"Task checkpoints are committed locally through "
                f"{head.stdout.strip()}; final tests passed."
            )
        return self._commit_paths(
            changed_paths,
            f"task {self.task_id}: verified completion",
            require_test=True,
        )


ACTIVE_CODE_TASK: Optional[CodeTaskWorkspace] = None


class RunTestsInput(BaseModel):
    pass


class CheckpointInput(BaseModel):
    label: str = Field(..., description="Short completed microtask/subtask checkpoint label.")
    paths: list[str] = Field(
        ...,
        description="Exact changed project-relative files to include, such as ['src/main.py', 'tests/test_main.py'].",
    )


class FinalizeInput(BaseModel):
    pass


@tool(args_schema=RunTestsInput)
def run_project_unit_tests() -> str:
    """Run the code task's unit tests in its isolated project directory."""
    if ACTIVE_CODE_TASK is None:
        return "Error: No isolated code task is active."
    return ACTIVE_CODE_TASK.run_tests()


@tool(args_schema=CheckpointInput)
def checkpoint_code_task(label: str, paths: list[str]) -> str:
    """Run project tests and create a local-only Git checkpoint for exact changed files."""
    if ACTIVE_CODE_TASK is None:
        return "Error: No isolated code task is active."
    return ACTIVE_CODE_TASK.checkpoint(label, paths)


@tool(args_schema=FinalizeInput)
def finalize_code_task() -> str:
    """Run final project tests and commit a verified local completion checkpoint."""
    if ACTIVE_CODE_TASK is None:
        return "Error: No isolated code task is active."
    return ACTIVE_CODE_TASK.finalize()
