"""Workspace-bounded file and directory tools."""

from langchain_core.tools import tool

from ..config import (
    BINARY_PROBE_BYTES,
    FILE_READ_CHUNK_BYTES,
    MAX_READ_FILE_BYTES,
)
from ..sandbox import SandboxManager, create_hitl_snapshot
from .schemas import EditFileInput, ListDirInput, ReadFileInput


@tool(args_schema=ReadFileInput)
def read_local_file(file_path: str) -> str:
    """Read a text file inside the validated workspace with configured size limits."""
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.exists():
            return f"Error: File '{file_path}' not found."

        max_size = MAX_READ_FILE_BYTES
        file_size = target.stat().st_size
        if file_size > max_size:
            return (
                f"Error: File '{file_path}' exceeds the maximum allowed size of "
                f"{max_size} bytes."
            )

        with target.open("rb") as source:
            header_bytes = source.read(BINARY_PROBE_BYTES)
            if b"\x00" in header_bytes:
                return (
                    f"Error: File '{file_path}' appears to be a binary file and "
                    "cannot be read as text."
                )

        chunks = []
        with target.open("rb") as source:
            while chunk := source.read(FILE_READ_CHUNK_BYTES):
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool(args_schema=EditFileInput)
def edit_local_file(file_path: str, content: str) -> str:
    """Create or overwrite a workspace file after creating a snapshot backup."""
    try:
        target = SandboxManager.validate_path(file_path)
        create_hitl_snapshot(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: File '{file_path}' written securely (HITL checkpoint snapshot created)."
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool(args_schema=ListDirInput)
def list_directory(dir_path: str = ".") -> str:
    """List files and subdirectories within the validated workspace."""
    try:
        target = SandboxManager.validate_path(dir_path)
        if not target.is_dir():
            return f"Error: '{dir_path}' is not a valid directory."
        items = []
        for child in target.iterdir():
            if child.name.startswith("."):
                continue
            prefix = "[DIR]  " if child.is_dir() else "[FILE] "
            relative_path = child.relative_to(SandboxManager.root_dir)
            items.append(f"{prefix} {relative_path}")
        return "\n".join(sorted(items)) if items else "Directory is empty."
    except Exception as exc:
        return f"Error listing directory: {exc}"
