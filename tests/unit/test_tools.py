import pytest
import asyncio
import subprocess
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


def test_research_plan_tool_output_marks_unvalidated_evidence_incomplete():
    import json

    from private_agent.tools.planner import _format_task

    formatted = json.loads(
        _format_task(
            {
                "task_id": "task-1",
                "task_type": "research",
                "revision": 2,
                "validation_revision": None,
                "validation_evidence": None,
                "plan": {
                    "research_questions": ["Which option is current?"],
                    "research_references": [],
                },
            }
        )
    )

    assert formatted["research_evidence_status"].startswith("incomplete")


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


@pytest.mark.asyncio
async def test_mcp_tool_errors_redact_sensitive_arguments_from_output_and_logs(
    monkeypatch,
):
    import private_agent.agent.runtime as agent

    secret = "private-tool-credential-value"
    tool = MagicMock()
    tool.name = "mcp_secret_failure"
    tool.ainvoke = AsyncMock(
        side_effect=RuntimeError(f"remote service echoed {secret}")
    )
    logger = MagicMock()
    output = []
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {tool.name: tool})
    monkeypatch.setattr(agent, "MCP_TOOL_NAMES", {tool.name})
    monkeypatch.setattr(agent, "RUN_LOGGER", logger)
    monkeypatch.setattr(
        agent.console,
        "print",
        lambda *values, **kwargs: output.append(" ".join(map(str, values))),
    )

    result = await execute_tool_call(
        {
            "name": tool.name,
            "args": {"nested": {"api_key": secret}},
            "id": "secret-error",
        },
        permission_mode="full",
    )

    logged_results = " ".join(str(call) for call in logger.info.call_args_list)
    assert "[REDACTED]" in result.content
    assert secret not in result.content
    assert secret not in logged_results
    assert secret not in "\n".join(output)


@pytest.mark.asyncio
async def test_mcp_tool_output_is_bounded_before_logging_and_provider_return(
    monkeypatch,
):
    import private_agent.agent.runtime as agent

    tool = MagicMock()
    tool.name = "mcp_large_output"
    tool.ainvoke = AsyncMock(return_value="x" * 100)
    logger = MagicMock()
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {tool.name: tool})
    monkeypatch.setattr(agent, "MCP_TOOL_NAMES", {tool.name})
    monkeypatch.setattr(agent, "RUN_LOGGER", logger)

    result = await execute_tool_call(
        {
            "name": tool.name,
            "args": {},
            "id": "large-output",
        },
        permission_mode="full",
        max_output_chars=12,
    )

    expected = "x" * 12 + "\n[Tool output truncated by configured limit.]"
    assert result.content == expected
    assert logger.info.call_args.args[-1] == expected
    assert len(logger.info.call_args.args[-1].split("\n", 1)[0]) == 12


@pytest.mark.asyncio
async def test_workspace_image_tool_cannot_be_used_in_online_mode(monkeypatch):
    import private_agent.agent.runtime as agent

    image_tool = MagicMock()
    image_tool.name = "load_workspace_image"
    image_tool.ainvoke = AsyncMock(return_value="unexpected local image")
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {image_tool.name: image_tool})
    monkeypatch.setattr(agent, "LOCAL_MEDIA_TOOL_NAMES", {image_tool.name})

    result = await execute_tool_call(
        {
            "name": image_tool.name,
            "args": {"file_path": "private.png"},
            "id": "online-image",
        },
        local_media_allowed=False,
        vision_supported=True,
        permission_mode="full",
    )

    image_tool.ainvoke.assert_not_awaited()
    assert "only in an explicitly selected local" in result.content


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


def test_workspace_write_rejects_racing_symlink_replacement(
    temp_workspace, tmp_path, monkeypatch
):
    from private_agent.tools import edit_local_file

    target = temp_workspace / "target.txt"
    target.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def replace_target(_file_fd, _name):
        target.unlink()
        target.symlink_to(outside)

    monkeypatch.setattr(
        "private_agent.tools.filesystem._snapshot_file_descriptor",
        replace_target,
    )
    result = edit_local_file.invoke(
        {"file_path": "target.txt", "content": "overwritten"}
    )

    assert "changed during the approved write" in result
    assert target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


def test_workspace_search_tools_find_files_and_text(temp_workspace):
    from private_agent.tools import search_workspace_files, search_workspace_text

    (temp_workspace / "src").mkdir()
    (temp_workspace / "src" / "main.py").write_text(
        "def answer():\n    return 'Juniper'\n", encoding="utf-8"
    )
    (temp_workspace / ".private").mkdir()
    (temp_workspace / ".private" / "hidden.py").write_text(
        "Juniper hidden", encoding="utf-8"
    )
    binary = temp_workspace / "binary.dat"
    binary.write_bytes(b"Juniper\x00binary")

    files = search_workspace_files.invoke({"pattern": "**/*.py"})
    matches = search_workspace_text.invoke(
        {"query": "juniper", "pattern": "*.py"}
    )

    assert "src/main.py" in files
    assert "hidden.py" not in files
    assert "src/main.py:2:" in matches
    assert "hidden.py" not in matches
    assert "binary.dat" not in matches


def test_workspace_symbol_search_finds_supported_declarations(temp_workspace):
    from private_agent.tools import search_workspace_symbols

    source_files = {
        "src/app.py": "class Widget:\n    async def build(self):\n        pass\n",
        "src/service.ts": (
            "export interface WidgetApi {}\n"
            "export async function buildWidget() {}\n"
            "export type WidgetOptions = {};\n"
        ),
        "src/server.go": "func BuildWidget() {}\ntype Widget struct {}\n",
        "src/lib.rs": "pub async fn build_widget() {}\npub struct Widget {}\n",
        "src/not-source.txt": "def Widget(): pass\n",
        ".private/hidden.py": "class Widget: pass\n",
    }
    for relative, contents in source_files.items():
        path = temp_workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    widget_results = search_workspace_symbols.invoke(
        {"symbol": "Widget", "pattern": "**/*"}
    )
    build_results = search_workspace_symbols.invoke(
        {"symbol": "buildWidget", "pattern": "**/*.ts"}
    )

    assert "src/app.py:1: class Widget:" in widget_results
    assert "src/service.ts:1: export interface WidgetApi" not in widget_results
    assert "src/server.go:2: type Widget struct" in widget_results
    assert "src/lib.rs:2: pub struct Widget" in widget_results
    assert "hidden.py" not in widget_results
    assert "not-source.txt" not in widget_results
    assert build_results == "src/service.ts:2: export async function buildWidget() {}"


def test_workspace_symbol_search_is_bounded_and_rejects_unsafe_patterns(
    temp_workspace, monkeypatch
):
    import private_agent.tools.filesystem as filesystem
    from private_agent.tools import search_workspace_symbols

    source = temp_workspace / "app.py"
    source.write_text(
        "".join(f"def repeated():\n    return {index}\n" for index in range(8)),
        encoding="utf-8",
    )
    result = search_workspace_symbols.invoke(
        {"symbol": "REPEATED", "pattern": "*.py", "limit": 2}
    )
    assert len(result.splitlines()) == 2
    assert "cannot traverse parents" in search_workspace_symbols.invoke(
        {"symbol": "repeated", "pattern": "../*.py"}
    )

    monkeypatch.setattr(filesystem, "_MAX_SEARCH_BYTES", 1)
    limited = search_workspace_symbols.invoke({"symbol": "repeated"})
    assert limited.startswith("No declarations found")
    assert "configured workspace scan limit" in limited


def test_workspace_symbol_search_skips_symlink_escape_and_invalid_utf8(
    temp_workspace, tmp_path
):
    from private_agent.tools import search_workspace_symbols

    external = tmp_path / "external.py"
    external.write_text("class PrivateType:\n    pass\n", encoding="utf-8")
    try:
        (temp_workspace / "external.py").symlink_to(external)
    except OSError:
        pytest.skip("Symlinks are unavailable")
    (temp_workspace / "invalid.py").write_bytes(b"class PrivateType:\xff\n")

    assert search_workspace_symbols.invoke(
        {"symbol": "PrivateType"}
    ) == "No declarations found for symbol 'PrivateType'."


def test_workspace_image_input_validates_path_and_size(temp_workspace, monkeypatch):
    from private_agent.config import MAX_CAPTURED_IMAGE_BYTES
    from private_agent.tools import load_workspace_image

    oversized = temp_workspace / "large.png"
    oversized.write_bytes(b"x" * (MAX_CAPTURED_IMAGE_BYTES + 1))
    assert "safety limit" in load_workspace_image.invoke(
        {"file_path": "large.png"}
    )
    assert "supported image file extension" in load_workspace_image.invoke(
        {"file_path": "notes.txt"}
    )

    outside = temp_workspace.parent / "outside.png"
    outside.write_bytes(b"image")
    assert "Security Violation" in load_workspace_image.invoke(
        {"file_path": str(outside)}
    )


def test_workspace_image_input_normalizes_and_consumes_single_use_image(
    temp_workspace, monkeypatch
):
    from private_agent.tools import load_workspace_image
    import private_agent.tools.media as media

    image_path = temp_workspace / "source.png"
    image_path.write_bytes(b"fake image bytes")
    frame = types.SimpleNamespace(shape=(2160, 3840, 3), ndim=3)
    encoded_bytes = b"normalized jpeg"

    class FakeEncodedImage:
        def tobytes(self):
            return encoded_bytes

    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.IMWRITE_JPEG_QUALITY = 2
    cv2.INTER_AREA = 3
    cv2.imdecode = MagicMock(return_value=frame)
    cv2.resize = MagicMock(return_value=frame)
    cv2.imencode = MagicMock(return_value=(True, FakeEncodedImage()))
    class FakeImageHeader:
        size = (3840, 2160)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(
        open=MagicMock(return_value=FakeImageHeader())
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(media, "_image_capture_count", 0)
    media._captured_images.clear()

    result = load_workspace_image.invoke({"file_path": "source.png"})
    reference = result.removeprefix("Image loaded in memory as ").split(".", 1)[0]

    assert reference.startswith("workspace-image:")
    assert "not copied or saved" in result
    assert cv2.resize.call_args.args[1] == (1280, 720)
    message = media.captured_image_message(reference)
    assert message.content[0]["text"] == "Analyze this user-approved local workspace image."
    assert "data:image/jpeg;base64," in message.content[1]["image_url"]["url"]
    assert media.captured_image_message(reference) is None


def test_workspace_image_rejects_excessive_decoded_dimensions(
    temp_workspace, monkeypatch
):
    from private_agent.config import MAX_DECODED_IMAGE_PIXELS
    from private_agent.tools import load_workspace_image

    image_path = temp_workspace / "large-dimensions.png"
    image_path.write_bytes(b"encoded data")
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.imdecode = MagicMock()
    class FakeImageHeader:
        size = (MAX_DECODED_IMAGE_PIXELS + 1, 1)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(
        open=MagicMock(return_value=FakeImageHeader())
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(
            frombuffer=MagicMock(return_value=b"encoded"),
            uint8=object(),
        ),
    )

    result = load_workspace_image.invoke({"file_path": "large-dimensions.png"})

    assert "pixel safety limit" in result
    cv2.imdecode.assert_not_called()


def test_workspace_search_tools_reject_traversal_and_symlink_escape(
    temp_workspace, tmp_path
):
    from private_agent.tools import search_workspace_files, search_workspace_text

    secret = tmp_path / "secret.py"
    secret.write_text("must stay outside", encoding="utf-8")
    try:
        (temp_workspace / "escape.py").symlink_to(secret)
    except OSError:
        pytest.skip("Symlinks are unavailable")

    assert "cannot traverse parents" in search_workspace_files.invoke(
        {"pattern": "../*.py"}
    )
    assert "cannot traverse parents" in search_workspace_text.invoke(
        {"query": "must", "pattern": "../*"}
    )
    assert "escape.py" not in search_workspace_files.invoke({"pattern": "*.py"})
    assert "must stay outside" not in search_workspace_text.invoke(
        {"query": "must"}
    )


def test_workspace_git_inspection_is_read_only_and_bounded(temp_workspace):
    from private_agent.tools import inspect_workspace_git

    subprocess.run(["git", "init", "-q"], cwd=temp_workspace, check=True)
    source = temp_workspace / "tracked.txt"
    source.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=temp_workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
            "-q",
        ],
        cwd=temp_workspace,
        check=True,
    )
    source.write_text("changed\n", encoding="utf-8")

    status = inspect_workspace_git.invoke({"action": "status"})
    diff = inspect_workspace_git.invoke({"action": "diff"})
    log = inspect_workspace_git.invoke({"action": "log"})
    branches = inspect_workspace_git.invoke({"action": "branches"})
    after_status = inspect_workspace_git.invoke({"action": "status"})

    assert "tracked.txt" in status
    assert "-initial" in diff
    assert "+changed" in diff
    assert "initial" in log
    assert "master" in branches or "main" in branches
    assert status == after_status


def test_workspace_patch_preview_matches_applied_patch(temp_workspace, monkeypatch):
    from private_agent.tools import apply_workspace_patch, preview_workspace_patch

    source = temp_workspace / "app.py"
    source.write_text("def answer():\n    return 1\n", encoding="utf-8")
    snapshots = []
    monkeypatch.setattr(
        "private_agent.tools.filesystem._snapshot_file_descriptor",
        lambda _file_fd, name, **_kwargs: snapshots.append(name),
    )
    arguments = {
        "file_path": "app.py",
        "old_text": "return 1",
        "new_text": "return 42",
    }

    preview = preview_workspace_patch.invoke(arguments)
    applied = apply_workspace_patch.invoke(arguments)

    assert "--- app.py" in preview
    assert "-    return 1" in preview
    assert "+    return 42" in preview
    assert preview in applied
    assert source.read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert snapshots == ["app.py"]


def test_workspace_patch_requires_unique_context_and_rejects_escape(temp_workspace):
    from private_agent.tools import apply_workspace_patch

    (temp_workspace / "duplicate.txt").write_text("same same", encoding="utf-8")
    ambiguous = apply_workspace_patch.invoke(
        {
            "file_path": "duplicate.txt",
            "old_text": "same",
            "new_text": "changed",
        }
    )
    escape = apply_workspace_patch.invoke(
        {
            "file_path": "../outside.txt",
            "old_text": "original",
            "new_text": "changed",
        }
    )

    assert "match exactly once" in ambiguous
    assert "Error applying patch" in escape
    assert (temp_workspace / "duplicate.txt").read_text(encoding="utf-8") == "same same"


def test_workspace_mutations_reject_symlinks_and_hard_linked_files(temp_workspace):
    from private_agent.tools import (
        apply_workspace_patch,
        delete_workspace_file,
        rename_workspace_file,
    )

    target = temp_workspace / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    symlink = temp_workspace / "link.txt"
    symlink.symlink_to(target)
    hard_link = temp_workspace / "hard-link.txt"
    hard_link.hardlink_to(target)

    patch_symlink = apply_workspace_patch.invoke(
        {"file_path": "link.txt", "old_text": "unchanged", "new_text": "changed"}
    )
    patch_hard_link = apply_workspace_patch.invoke(
        {
            "file_path": "hard-link.txt",
            "old_text": "unchanged",
            "new_text": "changed",
        }
    )
    rename_symlink = rename_workspace_file.invoke(
        {"source_path": "link.txt", "destination_path": "renamed-link.txt"}
    )
    delete_symlink = delete_workspace_file.invoke({"file_path": "link.txt"})

    assert "Symlink paths" in patch_symlink
    assert "multiple hard links" in patch_hard_link
    assert "Symlink paths" in rename_symlink
    assert "Symlink paths" in delete_symlink
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert symlink.is_symlink()


def test_workspace_patch_rejects_path_replacement_during_snapshot(
    temp_workspace, tmp_path, monkeypatch
):
    from private_agent.tools import apply_workspace_patch

    target = temp_workspace / "target.txt"
    target.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def replace_target(_file_fd, _name):
        target.unlink()
        target.symlink_to(outside)

    monkeypatch.setattr(
        "private_agent.tools.filesystem._snapshot_file_descriptor",
        lambda file_fd, name, **_kwargs: replace_target(file_fd, name),
    )
    result = apply_workspace_patch.invoke(
        {"file_path": "target.txt", "old_text": "inside", "new_text": "changed"}
    )

    assert "Workspace file changed during the approved write" in result
    assert target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


def test_workspace_mutations_fail_closed_without_secure_os_support(
    temp_workspace, monkeypatch
):
    from private_agent.tools import apply_workspace_patch

    target = temp_workspace / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    monkeypatch.setattr(
        "private_agent.tools.filesystem._secure_mutation_available",
        lambda: False,
    )

    result = apply_workspace_patch.invoke(
        {"file_path": "target.txt", "old_text": "unchanged", "new_text": "changed"}
    )

    assert "Secure workspace mutations require" in result
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_workspace_rename_does_not_overwrite_racing_destination(
    temp_workspace, monkeypatch
):
    from private_agent.tools import rename_workspace_file
    import private_agent.tools.filesystem as filesystem

    source = temp_workspace / "source.txt"
    source.write_text("source", encoding="utf-8")
    destination = temp_workspace / "destination.txt"
    original_link = filesystem.os.link

    def race_destination(*args, **kwargs):
        destination.write_text("keep destination", encoding="utf-8")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(filesystem.os, "link", race_destination)
    monkeypatch.setattr(
        filesystem.os,
        "supports_dir_fd",
        filesystem.os.supports_dir_fd | {race_destination},
    )
    result = rename_workspace_file.invoke(
        {"source_path": "source.txt", "destination_path": "destination.txt"}
    )

    assert "File exists" in result
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "keep destination"


@pytest.mark.asyncio
async def test_workspace_mutations_require_runtime_approval(temp_workspace, monkeypatch):
    import private_agent.agent.runtime as agent

    source = temp_workspace / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(
        "private_agent.tools.filesystem._snapshot_file_descriptor",
        lambda _file_fd, _name, **_kwargs: None,
    )
    arguments = {
        "name": "apply_workspace_patch",
        "args": {
            "file_path": "app.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
        },
        "id": "patch-approval",
    }

    monkeypatch.setattr(agent.console, "input", lambda _prompt: "n")
    denied = await execute_tool_call(arguments, permission_mode="auto")
    assert "declined" in denied.content
    assert source.read_text(encoding="utf-8") == "value = 1\n"

    monkeypatch.setattr(agent.console, "input", lambda _prompt: "y")
    approved = await execute_tool_call(arguments, permission_mode="auto")
    assert "Success: Applied patch" in approved.content
    assert source.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.asyncio
async def test_workspace_rename_and_delete_require_runtime_approval(
    temp_workspace, monkeypatch
):
    import private_agent.agent.runtime as agent

    source = temp_workspace / "source.txt"
    source.write_text("recoverable content", encoding="utf-8")
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    responses = iter(["n", "y", "n", "y"])
    monkeypatch.setattr(agent.console, "input", lambda _prompt: next(responses))

    rename_call = {
        "name": "rename_workspace_file",
        "args": {"source_path": "source.txt", "destination_path": "renamed.txt"},
        "id": "rename-approval",
    }
    denied_rename = await execute_tool_call(rename_call, permission_mode="auto")
    assert "declined" in denied_rename.content
    assert source.exists()

    approved_rename = await execute_tool_call(rename_call, permission_mode="auto")
    destination = temp_workspace / "renamed.txt"
    assert "Success" in approved_rename.content
    assert destination.read_text(encoding="utf-8") == "recoverable content"
    assert not source.exists()

    delete_call = {
        "name": "delete_workspace_file",
        "args": {"file_path": "renamed.txt"},
        "id": "delete-approval",
    }
    denied_delete = await execute_tool_call(delete_call, permission_mode="auto")
    assert "declined" in denied_delete.content
    assert destination.exists()

    approved_delete = await execute_tool_call(delete_call, permission_mode="auto")
    assert "recoverable snapshot" in approved_delete.content
    assert not destination.exists()
    snapshot_dir = temp_workspace / ".agent_snapshots"
    snapshots = [
        *snapshot_dir.glob("source.txt_*.bak"),
        *snapshot_dir.glob("renamed.txt_*.bak"),
    ]
    assert len(snapshots) == 2
    assert all(path.read_text(encoding="utf-8") == "recoverable content" for path in snapshots)


def test_workspace_rename_and_delete_are_recoverable_and_collision_safe(
    temp_workspace, monkeypatch
):
    from private_agent.tools import delete_workspace_file, rename_workspace_file

    source = temp_workspace / "source.txt"
    source.write_text("recoverable", encoding="utf-8")
    (temp_workspace / "existing.txt").write_text("keep", encoding="utf-8")
    snapshots = []
    monkeypatch.setattr(
        "private_agent.tools.filesystem._snapshot_file_descriptor",
        lambda _file_fd, name: snapshots.append(name),
    )

    collision = rename_workspace_file.invoke(
        {"source_path": "source.txt", "destination_path": "existing.txt"}
    )
    renamed = rename_workspace_file.invoke(
        {"source_path": "source.txt", "destination_path": "nested/renamed.txt"}
    )
    deleted = delete_workspace_file.invoke({"file_path": "nested/renamed.txt"})

    assert "already exists" in collision
    assert "Success" in renamed
    assert not source.exists()
    assert (temp_workspace / "existing.txt").read_text(encoding="utf-8") == "keep"
    assert "recoverable snapshot" in deleted
    assert not (temp_workspace / "nested" / "renamed.txt").exists()
    assert snapshots == ["source.txt", "renamed.txt"]


def test_file_snapshots_are_unique_and_workspace_local(temp_workspace):
    from private_agent.sandbox import create_hitl_snapshot

    source = temp_workspace / "important.txt"
    source.write_text("initial", encoding="utf-8")

    create_hitl_snapshot(source)
    create_hitl_snapshot(source)

    snapshot_directory = temp_workspace / ".agent_snapshots"
    snapshots = list(snapshot_directory.glob("important.txt_*.bak"))
    assert len(snapshots) == 2
    assert all(path.read_text(encoding="utf-8") == "initial" for path in snapshots)
    assert all(path.resolve().is_relative_to(temp_workspace) for path in snapshots)


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


def test_workspace_symbol_tool_is_registered_as_read_only():
    import private_agent.tools as tools

    assert tools.AVAILABLE_TOOLS["search_workspace_symbols"] is (
        tools.search_workspace_symbols
    )
    entry = next(
        entry
        for entry in describe_tool_catalog()
        if entry["name"] == "search_workspace_symbols"
    )
    assert entry["permission"] == "workspace-bounded read-only search"
    assert entry["effects"] == "read-only/local"


def test_workspace_media_tools_are_registered_as_local_approved_tools():
    import private_agent.tools as tools

    for name in ("load_workspace_video", "transcribe_workspace_audio"):
        assert name in tools.AVAILABLE_TOOLS
        assert name in tools.LOCAL_MEDIA_TOOL_NAMES

    catalog = {
        entry["name"]: entry
        for entry in describe_tool_catalog()
    }
    assert catalog["load_workspace_video"]["effects"] == (
        "samples an explicitly approved workspace video"
    )
    assert catalog["transcribe_workspace_audio"]["effects"] == (
        "transcribes an explicitly approved workspace audio file locally"
    )


def test_unavailable_ollama_returns_no_models_without_raising(monkeypatch):
    import private_agent.agent.runtime as agent

    class UnavailableClient:
        closed = False

        def __init__(self, **_kwargs):
            pass

        def list(self):
            raise ConnectionError("Ollama is not running")

        def close(self):
            self.closed = True

    client = UnavailableClient()
    monkeypatch.setattr(agent.ollama, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(agent.console, "print", lambda *_args, **_kwargs: None)

    assert agent.fetch_local_chat_models() == []
    assert client.closed


def test_agent_current_datetime_context_includes_local_date_and_timezone():
    from datetime import datetime
    from private_agent.agent import _current_datetime_context

    context = _current_datetime_context()
    now = datetime.now().astimezone()

    assert f"{now:%A, %B} {now.day}, {now.year}" in context
    assert f"{now:%H:%M:%S %Z} (UTC{now:%z})" in context
    assert "authoritative current date/time" in context

def test_tool_catalog_compactly_separates_names_descriptions_and_boundaries():
    import io
    from rich.console import Console as RichConsole
    from private_agent.agent import _print_tool_catalog

    output = io.StringIO()
    console = RichConsole(file=output, width=100, force_terminal=False)
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
    rendered = output.getvalue()
    assert "Tool Catalog" in rendered
    assert "Tool / source" in rendered
    assert "Description" in rendered
    assert "Permission / effects" in rendered
    for entry in entries:
        assert entry["name"] in rendered
        assert entry["description"].split()[0] in rendered
        assert entry["origin"] in rendered
        assert entry["permission"] in rendered
        assert entry["effects"] in rendered
    assert len(rendered.splitlines()) < 10


def test_override_tool_list_includes_platform_filtered_tools():
    from private_agent.agent import runtime

    tool = MagicMock()
    with (
        patch.object(runtime, "AVAILABLE_TOOLS", {"run_shell_command": tool}),
        patch.object(runtime, "OVERRIDE_TOOL_LIST", frozenset()),
    ):
        assert runtime._tools_for_model(
            provider_type="local",
            capabilities={"vision": True},
            isolation_warning="sandbox unavailable",
        ) == []

    with (
        patch.object(runtime, "AVAILABLE_TOOLS", {"run_shell_command": tool}),
        patch.object(runtime, "OVERRIDE_TOOL_LIST", {"run_shell_command"}),
    ):
        assert runtime._tools_for_model(
            provider_type="local",
            capabilities={"vision": True},
            isolation_warning="sandbox unavailable",
        ) == [tool]


def test_tool_catalog_marks_overridden_unavailable_entries_red_and_struck_out():
    import io
    from rich.console import Console as RichConsole
    from private_agent.agent import _print_tool_catalog

    output = io.StringIO()
    console = RichConsole(
        file=output,
        width=220,
        force_terminal=True,
        color_system="standard",
    )
    entry = {
        "name": "run_shell_command",
        "origin": "built-in",
        "permission": "approval required",
        "effects": "may modify local state",
        "description": "Runs isolated commands.",
    }
    with patch("private_agent.agent.runtime.console", console):
        _print_tool_catalog(
            [entry],
            unavailable_names={"run_shell_command"},
            override_tool_list={"run_shell_command"},
        )

    rendered = output.getvalue()
    assert "run_shell_command (built-in) (override)" in rendered
    assert "\x1b[9;31m" in rendered

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
    with pytest.raises(ValueError, match="unsupported transport"):
        await load_configured_mcp_tools(
            "websocket-timeout",
            {
                "transport": "websocket",
                "url": "wss://example.test/ws",
                "timeout": 5,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "url"),
    [
        ("sse", "https://service.example.test/sse"),
        ("streamable_http", "https://service.example.test/mcp"),
    ],
)
async def test_configured_mcp_http_transports_discover_tools(
    monkeypatch, transport, url
):
    captured = {}
    discovered_tool = MagicMock(name="remote_read")

    class FakeMCPClient:
        def __init__(self, configs):
            captured["config"] = configs["remote"]

        async def get_tools(self, server_name):
            captured["server_name"] = server_name
            return [discovered_tool]

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FakeMCPClient
    monkeypatch.setattr(
        "private_agent.tools.mcp.validate_explicit_service_url",
        lambda supplied_url, _schemes: supplied_url,
    )
    with patch.dict(
        sys.modules,
        {
            "langchain_mcp_adapters": parent,
            "langchain_mcp_adapters.client": adapter,
        },
    ):
        config = {"transport": transport, "url": url}
        config["timeout"] = 5
        tools = await load_configured_mcp_tools(
            "remote", config
        )

    assert tools == [discovered_tool]
    assert captured["server_name"] == "remote"
    assert captured["config"]["url"] == url
    assert captured["config"]["timeout"] == 5
    assert ("httpx_client_factory" in captured["config"]) == (
        transport in {"sse", "streamable_http"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "endpoint"),
    [
        ("sse", "/sse"),
        ("streamable_http", "/mcp"),
    ],
)
async def test_live_mcp_transports_close_connections_after_discovery_and_calls(
    monkeypatch, transport, endpoint
):
    import socket

    import uvicorn
    from mcp.server.fastmcp import FastMCP
    import private_agent.tools.mcp as mcp_module
    from private_agent.tools.network import mcp_http_client_factory

    server_app = FastMCP("private-agent-lifecycle-test")

    @server_app.tool()
    def add_one(value: int) -> int:
        return value + 1

    if transport == "sse":
        app = server_app.sse_app()
    else:
        app = server_app.streamable_http_app()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    url = f"http://127.0.0.1:{listener.getsockname()[1]}{endpoint}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    clients = []

    def tracking_client_factory(**kwargs):
        client = mcp_http_client_factory(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(mcp_module, "mcp_http_client_factory", tracking_client_factory)

    try:
        for _ in range(250):
            if server.started:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.02)
        assert server.started, "local MCP test server did not start"

        for _ in range(2):
            config = {"transport": transport, "url": url}
            config["timeout"] = 5
            tools = await load_configured_mcp_tools(
                "local", config
            )
            assert [tool.name for tool in tools] == ["add_one"]
            assert clients and all(client.is_closed for client in clients)

            result = await tools[0].ainvoke({"value": 2})
            assert any(part.get("text") == "3" for part in result)
            assert all(client.is_closed for client in clients)
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


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


@pytest.mark.asyncio
async def test_mcp_stdio_sandbox_is_removed_when_client_initialization_fails(
    monkeypatch, tmp_path
):
    import private_agent.tools.mcp as mcp_module

    sandbox_dir = tmp_path / "mcp-sandbox"

    def make_sandbox(**_kwargs):
        sandbox_dir.mkdir()
        return str(sandbox_dir)

    class FailingMCPClient:
        def __init__(self, _configs):
            raise RuntimeError("client startup failure")

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FailingMCPClient
    monkeypatch.setattr(mcp_module.tempfile, "mkdtemp", make_sandbox)
    monkeypatch.setattr(
        "private_agent.code_tasks._isolated_command",
        lambda command, root: (["/usr/bin/bwrap", *command], {}),
    )
    try:
        with patch.dict(
            sys.modules,
            {
                "langchain_mcp_adapters": parent,
                "langchain_mcp_adapters.client": adapter,
            },
        ):
            with pytest.raises(RuntimeError, match="Failed to load MCP server"):
                await load_configured_mcp_tools(
                    "broken-stdio",
                    {"transport": "stdio", "command": "python", "args": ["server.py"]},
                )
        assert not sandbox_dir.exists()
        assert str(sandbox_dir) not in mcp_module._mcp_sandbox_dirs
    finally:
        close_mcp_sandbox_dirs()


@pytest.mark.asyncio
async def test_mcp_stdio_sandbox_is_removed_when_tool_discovery_fails(
    monkeypatch, tmp_path
):
    import private_agent.tools.mcp as mcp_module

    sandbox_dir = tmp_path / "mcp-discovery-failure"

    def make_sandbox(**_kwargs):
        sandbox_dir.mkdir()
        return str(sandbox_dir)

    class FailingDiscoveryClient:
        def __init__(self, _configs):
            pass

        async def get_tools(self, server_name):
            raise RuntimeError(f"{server_name} process exited")

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FailingDiscoveryClient
    monkeypatch.setattr(mcp_module.tempfile, "mkdtemp", make_sandbox)
    monkeypatch.setattr(
        "private_agent.code_tasks._isolated_command",
        lambda command, root: (["/usr/bin/bwrap", *command], {}),
    )
    try:
        with patch.dict(
            sys.modules,
            {
                "langchain_mcp_adapters": parent,
                "langchain_mcp_adapters.client": adapter,
            },
        ):
            with pytest.raises(RuntimeError, match="Failed to load MCP server"):
                await load_configured_mcp_tools(
                    "failed-discovery",
                    {"transport": "stdio", "command": "python", "args": ["server.py"]},
                )
        assert not sandbox_dir.exists()
        assert str(sandbox_dir) not in mcp_module._mcp_sandbox_dirs
    finally:
        close_mcp_sandbox_dirs()


def test_mcp_cleanup_continues_after_one_sandbox_removal_fails(monkeypatch, tmp_path):
    import private_agent.tools.mcp as mcp_module

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(mcp_module, "_mcp_sandbox_dirs", [str(first), str(second)])
    real_rmtree = mcp_module.shutil.rmtree

    def failing_first(path, *args, **kwargs):
        if path == str(first):
            raise OSError("simulated cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(mcp_module.shutil, "rmtree", failing_first)

    close_mcp_sandbox_dirs()

    assert first.exists()
    assert not second.exists()
    assert mcp_module._mcp_sandbox_dirs == [str(first)]
    mcp_module._remove_mcp_sandbox_dir(str(first))


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
