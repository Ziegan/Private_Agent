"""Built-in web search, fetch, and download tools."""

import asyncio
import os
import tempfile
from langchain_core.tools import tool
from ..config import (
    MAX_DOWNLOAD_BYTES,
    MAX_FETCHED_WEBPAGE_CHARS,
    MAX_WEBPAGE_BYTES,
    NETWORK_REQUEST_TIMEOUT,
)
from ..sandbox import SandboxManager
from . import _search_duckduckgo, has_internet_connection, require_internet
from .network import _get_limited_response, _public_only_async_client
from .schemas import DownloadWebFileInput, FetchWebpageInput, WebSearchInput


@tool(args_schema=WebSearchInput)
@require_internet
def web_search(query: str) -> str:
    """Search the web for up-to-date info, technical documentation, or code references."""
    try:
        return _search_duckduckgo(query)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


@tool(args_schema=FetchWebpageInput)
@require_internet
def fetch_webpage(url: str) -> str:
    """Fetch, clean, and read article or documentation text from a URL."""

    async def _async_fetch():
        from bs4 import BeautifulSoup

        async with _public_only_async_client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=NETWORK_REQUEST_TIMEOUT,
        ) as client:
            body, encoding = await _get_limited_response(
                client,
                url,
                MAX_WEBPAGE_BYTES,
            )
        soup = BeautifulSoup(body.decode(encoding, errors="replace"), "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return (
            text[:MAX_FETCHED_WEBPAGE_CHARS] + "\n[Truncated...]"
            if len(text) > MAX_FETCHED_WEBPAGE_CHARS
            else text
        )

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _async_fetch()).result()
        return asyncio.run(_async_fetch())
    except ImportError:
        return "Failed to fetch webpage: Required package 'httpx' or 'beautifulsoup4' is not installed."
    except Exception as exc:
        return f"Failed to fetch webpage: {exc}"


@tool(args_schema=DownloadWebFileInput)
def download_web_file(url: str, save_path: str) -> str:
    """Download a file securely inside the workspace."""
    try:
        target = SandboxManager.validate_path(save_path)
    except Exception as exc:
        return f"Failed to download file: {exc}"
    if target.suffix.lower() in {".sh", ".exe", ".bat", ".cmd", ".msi"}:
        return (
            f"Error: Downloading executable/script extension '{target.suffix}' "
            "is restricted by security policy."
        )
    if not has_internet_connection():
        return "Error: Network unavailable. Please check your internet connection and try again."

    async def _async_download():
        target = SandboxManager.validate_path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            async with _public_only_async_client(
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                timeout=NETWORK_REQUEST_TIMEOUT,
            ) as client:
                body, _ = await _get_limited_response(
                    client,
                    url,
                    MAX_DOWNLOAD_BYTES,
                )
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = output.name
                output.write(body)
            os.replace(temporary_path, target)
        except Exception:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
            raise
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
        return asyncio.run(_async_download())
    except Exception as exc:
        return f"Failed to download file: {exc}"
