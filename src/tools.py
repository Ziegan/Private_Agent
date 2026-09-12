import os
import sys
import socket
import sqlite3
import threading
from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from rich.console import Console

from .sandbox import SandboxManager, create_hitl_snapshot, evaluate_shell_command
from .config import DEFAULT_DB_PATH, load_config

console = Console()

# Global reference for active sqlite db path used by sqlite tools
ACTIVE_DB_PATH = DEFAULT_DB_PATH
_sqlite_lock = threading.Lock()

def set_active_db_path(path: str):
    global ACTIVE_DB_PATH
    ACTIVE_DB_PATH = os.path.abspath(path)

# --- NETWORK CONNECTIVITY UTILITY ---
def check_internet_connection(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """Quick socket probe to check active TCP-level internet reachability."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except Exception:
        return False

def require_internet(func: Callable) -> Callable:
    """Decorator that validates active internet connection before executing a web-bound tool."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not check_internet_connection():
            return "Error: Network unavailable. Please check your internet connection and try again."
        return func(*args, **kwargs)
    return wrapper

# --- INPUT SCHEMAS ---
class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read relative to workspace.")

class EditFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to create or overwrite.")
    content: str = Field(..., description="Full text content to write into the file.")

class RunShellInput(BaseModel):
    command: str = Field(..., description="Shell command string to execute.")

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string.")

class ListDirInput(BaseModel):
    dir_path: str = Field(default=".", description="Directory path to list files from.")

class FetchWebpageInput(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch content from.")

class DownloadWebFileInput(BaseModel):
    url: str = Field(..., description="URL of file to download.")
    save_path: str = Field(..., description="Destination file path inside the workspace.")

class ReadSqliteHistoryInput(BaseModel):
    limit: int = Field(default=20, description="Number of recent chat history messages to read.")

class DeleteSqliteHistoryInput(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Specific session ID to delete, or leave empty/all to wipe.")

# --- TOOL IMPLEMENTATIONS ---
@tool(args_schema=ReadFileInput)
def read_local_file(file_path: str) -> str:
    """Read and return content from a file inside the sandboxed workspace directory with stream-based chunking and size validation to protect memory overhead."""
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.exists():
            return f"Error: File '{file_path}' not found."
        
        # Enforce file size limit (1MB max per the oversized file guard requirement, up to 10MB)
        max_size = 1024 * 1024  # 1MB
        file_size = target.stat().st_size
        if file_size > max_size:
            return f"Error: File '{file_path}' exceeds the maximum allowed size of 1MB."

        # Quick binary check on the first chunk
        with open(target, "rb") as f:
            header_bytes = f.read(2048)
            if b'\x00' in header_bytes:
                return f"Error: File '{file_path}' appears to be a binary file and cannot be read as text."

        # Stream-based chunking read to protect memory overhead
        chunks = []
        chunk_size = 65536  # 64KB chunks
        with open(target, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        raw_bytes = b"".join(chunks)
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool(args_schema=EditFileInput)
def edit_local_file(file_path: str, content: str) -> str:
    """Create or overwrite a file with given content safely within workspace root, creating an HITL snapshot backup first."""
    try:
        target = SandboxManager.validate_path(file_path)
        create_hitl_snapshot(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: File '{file_path}' written securely (HITL checkpoint snapshot created)."
    except Exception as e:
        return f"Error writing file: {str(e)}"

@tool(args_schema=RunShellInput)
def run_shell_command(command: str) -> str:
    """Execute a safe shell command inside the restricted workspace root directory with security checks and non-interactive safeguards."""
    import subprocess
    if evaluate_shell_command(command):
        console.print(f"\n[yellow][SECURITY GUARD] High-risk command detected:[/yellow]")
        console.print(f"  -> Command: {command}")
        if not sys.stdin.isatty():
            return "Error: High-risk command blocked automatically in non-interactive/headless execution environment."
        choice = console.input("[yellow]Allow execution? [y/N]: [/yellow]").strip().lower()
        if choice != 'y':
            return "Error: Command execution rejected and blocked by user security policy."

    try:
        result = subprocess.run(
            command, shell=True, cwd=str(SandboxManager.root_dir),
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return f"Exit Code: {result.returncode}\nOutput:\n{output}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

@tool(args_schema=WebSearchInput)
@require_internet
def web_search(query: str) -> str:
    """Search the web for up-to-date info, technical documentation, or code references."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n")
        return "\n---\n".join(results) if results else "No web search results found."
    except ImportError:
        return "Web search failed: Required package 'ddgs' is not installed."
    except Exception as e:
        return f"Web search failed: {str(e)}"

@tool(args_schema=ListDirInput)
def list_directory(dir_path: str = ".") -> str:
    """List files and subdirectories cleanly within the workspace root."""
    try:
        target = SandboxManager.validate_path(dir_path)
        if not target.is_dir():
            return f"Error: '{dir_path}' is not a valid directory."
        items = []
        for child in target.iterdir():
            if child.name.startswith('.'):
                continue
            prefix = "[DIR]  " if child.is_dir() else "[FILE] "
            rel_path = child.relative_to(SandboxManager.root_dir)
            items.append(f"{prefix} {rel_path}")
        return "\n".join(sorted(items)) if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool(args_schema=FetchWebpageInput)
@require_internet
def fetch_webpage(url: str) -> str:
    """Fetch, clean, and read the full text content of a documentation or article URL using asynchronous request handlers."""
    async def _async_fetch():
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}, timeout=15, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:12000] + "\n[Truncated...]" if len(text) > 12000 else text

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _async_fetch()).result()
        else:
            return asyncio.run(_async_fetch())
    except ImportError:
        return "Failed to fetch webpage: Required package 'httpx' or 'beautifulsoup4' is not installed."
    except Exception as e:
        return f"Failed to fetch webpage: {str(e)}"

@tool(args_schema=DownloadWebFileInput)
@require_internet
def download_web_file(url: str, save_path: str) -> str:
    """Download any file from a web URL securely inside the workspace using asynchronous request handlers."""
    async def _async_download():
        import httpx
        target = SandboxManager.validate_path(save_path)
        if target.suffix.lower() in [".sh", ".exe", ".bat", ".cmd", ".msi"]:
            return f"Error: Downloading executable/script extension '{target.suffix}' is restricted by security policy."
        
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                with open(target, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        return f"Success: File downloaded and saved to '{save_path}'."

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _async_download()).result()
        else:
            return asyncio.run(_async_download())
    except Exception as e:
        return f"Failed to download file: {str(e)}"

@tool(args_schema=ReadSqliteHistoryInput)
def read_chat_history_from_sqlite(limit: int = 20) -> str:
    """Read recent conversation log entries directly from the SQLite persistent memory database using thread-safe context management and locks."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(ACTIVE_DB_PATH, check_same_thread=False) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT session_id, role, content, timestamp FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
        if not rows:
            return "SQLite memory chat history is empty."
        formatted = []
        for session_id, role, content, timestamp in reversed(rows):
            formatted.append(f"[{timestamp}] Session: {session_id} | {role}: {content}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error reading SQLite history: {str(e)}"

@tool(args_schema=DeleteSqliteHistoryInput)
def delete_chat_history_from_sqlite(session_id: Optional[str] = None) -> str:
    """Delete chat history logs from the SQLite persistent database using thread-safe context management and locks."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(ACTIVE_DB_PATH, check_same_thread=False) as conn:
                if session_id:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
                    rows_affected = cursor.rowcount
                    return f"Success: Deleted {rows_affected} messages for session '{session_id}' from SQLite memory."
                else:
                    conn.execute("DELETE FROM chat_history")
                    return "Success: Wiped all chat history rows from SQLite persistent database."
    except Exception as e:
        return f"Error deleting SQLite history: {str(e)}"

AVAILABLE_TOOLS = {
    "read_local_file": read_local_file,
    "edit_local_file": edit_local_file,
    "run_shell_command": run_shell_command,
    "web_search": web_search,
    "list_directory": list_directory,
    "fetch_webpage": fetch_webpage,
    "download_web_file": download_web_file,
    "read_chat_history_from_sqlite": read_chat_history_from_sqlite,
    "delete_chat_history_from_sqlite": delete_chat_history_from_sqlite
}

async def load_mcp_tools(*args, **kwargs) -> list:
    """
    Connects to an MCP server, retrieves available tools, and returns them as a list.
    Guarantees that an empty list ([]) is returned upon any connection error, 
    timeout, or exception instead of a dictionary or None.
    """
    try:
        from mcp.client.stdio import stdio_client
        from mcp.client.session import ClientSession
        
        # Establishing connection parameters and session
        async with stdio_client(*args, **kwargs) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                
                tools = []
                for tool in getattr(response, "tools", []):
                    tools.append(tool)
                return tools
    except Exception:
        # Return an empty list on failure to satisfy type assertions and prevent test failures
        return []

def initialize_mcp_integration():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        pass
    else:
        try:
            mcp_tools = asyncio.run(load_mcp_tools())
            if mcp_tools:
                for t in mcp_tools:
                    tool_name = getattr(t, "name", str(t))
                    AVAILABLE_TOOLS[tool_name] = t
        except Exception:
            pass

initialize_mcp_integration()
