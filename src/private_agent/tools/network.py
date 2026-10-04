"""Peer-validated HTTP transports and managed outbound HTTP clients."""

import asyncio
import ipaddress
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse

import httpcore
import httpx
from bs4 import BeautifulSoup
from rich.console import Console

from ..config import (
    MAX_HTTP_REDIRECTS,
    MAX_NETWORK_CONCURRENCY,
    MAX_SEARCH_QUERY_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_STREAM_TIMEOUT,
    MAX_WEBPAGE_BYTES,
    NETWORK_REQUEST_TIMEOUT,
)

console = Console()
_outbound_http_clients = []
_EXPLICIT_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)

_RESOLVER_SCRIPT = (
    "import json,socket,sys;"
    "records=socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),0,socket.SOCK_STREAM);"
    "print(json.dumps([record[4][0] for record in records]))"
)


def _resolver_command(host: str, port: int):
    return [sys.executable, "-I", "-S", "-c", _RESOLVER_SCRIPT, host, str(port)]


def _resolver_environment():
    return {"LANG": "C", "PATH": os.defpath}


def _parse_resolver_output(output: str):
    try:
        addresses = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OSError("Outbound resolver returned an invalid response.") from exc
    if not isinstance(addresses, list) or not all(
        isinstance(address, str) for address in addresses
    ):
        raise OSError("Outbound resolver returned an invalid response.")
    return addresses


def _getaddrinfo_with_timeout(host: str, port: int, timeout: float):
    """Resolve in a killable child so stalled system DNS cannot outlive its deadline."""
    try:
        result = subprocess.run(
            _resolver_command(host, port),
            capture_output=True,
            check=True,
            close_fds=True,
            env=_resolver_environment(),
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise httpcore.ConnectTimeout("Outbound DNS resolution timed out.") from exc
    except subprocess.CalledProcessError as exc:
        raise OSError("Outbound hostname could not be resolved.") from exc
    return _parse_resolver_output(result.stdout)


async def _getaddrinfo_async_with_timeout(host: str, port: int, timeout: float):
    """Resolve asynchronously and kill/reap the resolver if cancelled or timed out."""
    process = await asyncio.create_subprocess_exec(
        *_resolver_command(host, port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_resolver_environment(),
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        await _kill_and_reap_resolver(process)
        raise httpcore.ConnectTimeout("Outbound DNS resolution timed out.") from exc
    except asyncio.CancelledError:
        await _kill_and_reap_resolver(process)
        raise
    if process.returncode:
        raise OSError("Outbound hostname could not be resolved.")
    return _parse_resolver_output(stdout.decode("utf-8", errors="strict"))


async def _kill_and_reap_resolver(process):
    try:
        process.kill()
    except ProcessLookupError:
        pass
    await process.communicate()


def _connect_deadline(timeout):
    if timeout is None:
        duration = NETWORK_REQUEST_TIMEOUT
    else:
        duration = min(max(0.0, float(timeout)), NETWORK_REQUEST_TIMEOUT)
    return time.monotonic() + duration


def _validated_peer_addresses(host: str, records, allow_loopback: bool):
    addresses = []
    explicit_local_host = host.lower() == "localhost"
    literal_address = None
    try:
        literal_address = ipaddress.ip_address(host.split("%", 1)[0])
        explicit_local_host = explicit_local_host or (
            literal_address.is_loopback
            or any(
                literal_address.version == network.version
                and literal_address in network
                for network in _EXPLICIT_PRIVATE_NETWORKS
            )
        )
    except ValueError:
        pass
    for record in records:
        resolved_address = record if isinstance(record, str) else record[4][0]
        address = ipaddress.ip_address(resolved_address.split("%", 1)[0])
        if not address.is_global and not (
            allow_loopback
            and explicit_local_host
            and (
                address.is_loopback
                if literal_address is None
                else address == literal_address
            )
        ):
            raise OSError(
                f"Outbound connection to non-public address {address} was blocked."
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise OSError("Outbound hostname resolved to no addresses.")
    return addresses


def _resolve_validated_peer(
    host: str,
    port: int,
    allow_loopback: bool,
    timeout: float = NETWORK_REQUEST_TIMEOUT,
):
    deadline = time.monotonic() + min(max(0.0, timeout), NETWORK_REQUEST_TIMEOUT)
    if deadline <= time.monotonic():
        raise httpcore.ConnectTimeout("Outbound DNS resolution timed out.")
    try:
        literal_address = ipaddress.ip_address(host.split("%", 1)[0])
        records = [str(literal_address)]
    except ValueError:
        try:
            records = _getaddrinfo_with_timeout(
                host,
                port,
                max(0.0, deadline - time.monotonic()),
            )
        except httpcore.ConnectTimeout as exc:
            raise OSError("Outbound DNS resolution timed out.") from exc
    return _validated_peer_addresses(host, records, allow_loopback)


async def _resolve_validated_peer_async(
    host: str,
    port: int,
    allow_loopback: bool,
    timeout: float,
):
    deadline = time.monotonic() + min(max(0.0, timeout), NETWORK_REQUEST_TIMEOUT)
    if deadline <= time.monotonic():
        raise httpcore.ConnectTimeout("Outbound DNS resolution timed out.")
    try:
        literal_address = ipaddress.ip_address(host.split("%", 1)[0])
        records = [str(literal_address)]
    except ValueError:
        records = await _getaddrinfo_async_with_timeout(
            host,
            port,
            max(0.0, deadline - time.monotonic()),
        )
    return _validated_peer_addresses(host, records, allow_loopback)


class _PublicOnlyNetworkBackend:
    """Resolve and validate each peer immediately before connecting to its IP."""

    def __init__(self, allow_loopback: bool = False):
        from httpcore._backends.auto import AutoBackend

        self._backend = AutoBackend()
        self._allow_loopback = allow_loopback

    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        deadline = _connect_deadline(timeout)
        addresses = await _resolve_validated_peer_async(
            host,
            port,
            self._allow_loopback,
            max(0.0, deadline - time.monotonic()),
        )
        last_error = None
        for address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise httpcore.ConnectTimeout("Outbound connection timed out.")
            try:
                return await self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=remaining,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (OSError, httpcore.ConnectError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("Outbound hostname resolved to no connectable addresses.")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("Unix-domain sockets are disabled for outbound web requests.")

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


class _PublicOnlySyncNetworkBackend:
    def __init__(self, allow_loopback: bool = False):
        from httpcore._backends.sync import SyncBackend

        self._backend = SyncBackend()
        self._allow_loopback = allow_loopback

    def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        deadline = _connect_deadline(timeout)
        addresses = _resolve_validated_peer(
            host,
            port,
            self._allow_loopback,
            max(0.0, deadline - time.monotonic()),
        )
        last_error = None
        for address in addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise httpcore.ConnectTimeout("Outbound connection timed out.")
            try:
                return self._backend.connect_tcp(
                    str(address),
                    port,
                    timeout=remaining,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (OSError, httpcore.ConnectError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("Outbound hostname resolved to no connectable addresses.")

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise OSError("Unix-domain sockets are disabled for outbound requests.")

    def sleep(self, seconds):
        return self._backend.sleep(seconds)


class _PublicOnlyAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    def __init__(self, allow_loopback: bool = False):
        limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
        super().__init__(trust_env=False, limits=limits, retries=0)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PublicOnlyNetworkBackend(allow_loopback),
        )


class _PublicOnlySyncHTTPTransport(httpx.HTTPTransport):
    def __init__(self, allow_loopback: bool = False):
        limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
        super().__init__(trust_env=False, limits=limits, retries=0)
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=self._pool._ssl_context,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            http1=True,
            http2=False,
            network_backend=_PublicOnlySyncNetworkBackend(allow_loopback),
        )


def _public_only_async_client(
    *,
    headers: dict,
    timeout: float,
    allow_loopback: bool = False,
    track: bool = False,
):
    client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(timeout),
        transport=_PublicOnlyAsyncHTTPTransport(allow_loopback),
        follow_redirects=False,
        trust_env=False,
    )
    if track:
        _outbound_http_clients.append(client)
    return client


def public_only_sync_client(
    *, headers: dict, timeout: float, allow_loopback=False, track: bool = True
):
    client = httpx.Client(
        headers=headers,
        timeout=timeout,
        transport=_PublicOnlySyncHTTPTransport(allow_loopback),
        follow_redirects=False,
        trust_env=False,
    )
    if track:
        _outbound_http_clients.append(client)
    return client


def ollama_client_kwargs() -> dict:
    """Return Ollama SDK kwargs using peer-validated transports."""
    return {
        "transport": _PublicOnlySyncHTTPTransport(allow_loopback=True),
        "timeout": httpx.Timeout(MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT),
        "follow_redirects": False,
        "trust_env": False,
    }


def ollama_langchain_client_kwargs() -> dict:
    """Return LangChain Ollama SDK custom sync/async HTTP client arguments."""
    return {
        "sync_client_kwargs": {
            "transport": _PublicOnlySyncHTTPTransport(allow_loopback=True),
            "timeout": httpx.Timeout(
                MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT
            ),
            "follow_redirects": False,
            "trust_env": False,
        },
        "async_client_kwargs": {
            "transport": _PublicOnlyAsyncHTTPTransport(allow_loopback=True),
            "timeout": httpx.Timeout(
                MAX_STREAM_TIMEOUT, connect=NETWORK_REQUEST_TIMEOUT
            ),
            "follow_redirects": False,
            "trust_env": False,
        },
    }


def track_ollama_http_clients(owner):
    """Track HTTPX clients created internally by Ollama SDK wrappers."""
    for attribute in ("_client", "_async_client"):
        client = getattr(owner, attribute, None)
        for _ in range(2):
            if isinstance(client, (httpx.Client, httpx.AsyncClient)):
                break
            client = getattr(client, "_client", None)
        if isinstance(client, (httpx.Client, httpx.AsyncClient)):
            if client not in _outbound_http_clients:
                _outbound_http_clients.append(client)


async def close_outbound_http_clients():
    clients, _outbound_http_clients[:] = list(_outbound_http_clients), []
    for client in clients:
        try:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
            else:
                await asyncio.to_thread(client.close)
        except Exception as exc:
            console.print(
                f"[yellow][Shutdown warning] Could not close an outbound "
                f"HTTP client: {type(exc).__name__}[/yellow]"
            )


def mcp_http_client_factory(headers=None, timeout=None, auth=None):
    limits = httpx.Limits(max_connections=MAX_NETWORK_CONCURRENCY)
    transport = _PublicOnlyAsyncHTTPTransport(allow_loopback=True)
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(NETWORK_REQUEST_TIMEOUT),
        auth=auth,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        limits=limits,
    )


_network_slots = threading.BoundedSemaphore(MAX_NETWORK_CONCURRENCY)


def validate_outbound_url(url: str) -> str:
    """Reject non-HTTP(S), credentialed, local, and non-public network targets."""
    try:
        normalized_url = _normalize_outbound_url(url)
        parsed = urllib.parse.urlsplit(normalized_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        _resolve_validated_peer(
            parsed.hostname,
            port,
            allow_loopback=False,
        )
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unsafe outbound URL: {exc}") from exc
    return normalized_url


def _normalize_outbound_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP(S) URLs with a host are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    hostname = parsed.hostname.rstrip(".")
    if not hostname:
        raise ValueError("The URL must contain a hostname.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


async def _validate_outbound_url_async(url: str) -> str:
    """Validate URL policy without leaving DNS work behind on cancellation."""
    try:
        normalized_url = _normalize_outbound_url(url)
        parsed = urllib.parse.urlsplit(normalized_url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        await _resolve_validated_peer_async(
            parsed.hostname,
            port,
            allow_loopback=False,
            timeout=NETWORK_REQUEST_TIMEOUT,
        )
    except (ValueError, OSError) as exc:
        raise ValueError(f"Unsafe outbound URL: {exc}") from exc
    return normalized_url


def validate_explicit_service_url(url: str, transports: set[str]) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in transports
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError(
            "Service URL must use an allowed scheme and contain no credentials or fragment."
        )
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("MCP service URLs must use HTTP or HTTPS.")
    _resolve_validated_peer(parsed.hostname, port, allow_loopback=True)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
    )


async def _get_limited_response(
    client, url: str, max_bytes: int
) -> tuple[bytes, str]:
    acquired = await asyncio.to_thread(
        _network_slots.acquire, True, NETWORK_REQUEST_TIMEOUT
    )
    if not acquired:
        raise TimeoutError("Outbound request concurrency limit wait timed out.")
    try:
        async def request_with_redirects():
            current_url = await _validate_outbound_url_async(url)
            for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_HTTP_REDIRECTS:
                            raise ValueError(
                                "HTTP redirect limit exceeded or redirect target missing."
                            )
                        current_url = await _validate_outbound_url_async(
                            urllib.parse.urljoin(current_url, location),
                        )
                        continue
                    response.raise_for_status()
                    declared_size = response.headers.get("content-length")
                    if declared_size and int(declared_size) > max_bytes:
                        raise ValueError(
                            f"Response exceeds the {max_bytes}-byte size limit."
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise ValueError(
                                f"Response exceeds the {max_bytes}-byte size limit."
                            )
                    return bytes(body), response.encoding or "utf-8"
            raise ValueError("HTTP redirect limit exceeded.")

        return await asyncio.wait_for(
            request_with_redirects(),
            timeout=NETWORK_REQUEST_TIMEOUT,
        )
    finally:
        _network_slots.release()


def search_duckduckgo(query: str, client_factory=public_only_sync_client) -> str:
    if not query.strip() or len(query) > MAX_SEARCH_QUERY_CHARS:
        raise ValueError(
            f"Search query must contain 1 to {MAX_SEARCH_QUERY_CHARS} characters."
        )
    acquired = _network_slots.acquire(timeout=NETWORK_REQUEST_TIMEOUT)
    if not acquired:
        raise TimeoutError("Outbound request concurrency limit wait timed out.")
    try:
        current_url = _normalize_outbound_url(
            "https://html.duckduckgo.com/html/"
        )
        with client_factory(
            headers={"User-Agent": "PrivateAgent/1.0"},
            timeout=httpx.Timeout(NETWORK_REQUEST_TIMEOUT),
            track=False,
        ) as client:
            for redirect_count in range(MAX_HTTP_REDIRECTS + 1):
                with client.stream(
                    "GET",
                    current_url,
                    params={"q": query} if redirect_count == 0 else None,
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count == MAX_HTTP_REDIRECTS:
                            raise ValueError(
                                "HTTP redirect limit exceeded or redirect target missing."
                            )
                        current_url = _normalize_outbound_url(
                            urllib.parse.urljoin(current_url, location)
                        )
                        continue
                    response.raise_for_status()
                    declared_size = response.headers.get("content-length")
                    if declared_size and int(declared_size) > MAX_WEBPAGE_BYTES:
                        raise ValueError(
                            f"Response exceeds the {MAX_WEBPAGE_BYTES}-byte size limit."
                        )
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_WEBPAGE_BYTES:
                            raise ValueError(
                                f"Response exceeds the {MAX_WEBPAGE_BYTES}-byte size limit."
                            )
                    soup = BeautifulSoup(
                        bytes(body).decode(
                            response.encoding or "utf-8", errors="replace"
                        ),
                        "html.parser",
                    )
                    results = []
                    for result in soup.select(".result")[:MAX_SEARCH_RESULTS]:
                        anchor = result.select_one("a.result__a")
                        if not anchor or not anchor.get("href"):
                            continue
                        snippet = result.select_one(".result__snippet")
                        results.append(
                            f"Title: {anchor.get_text(' ', strip=True)}\n"
                            f"Snippet: {snippet.get_text(' ', strip=True) if snippet else ''}\n"
                            f"URL: {anchor['href']}"
                        )
                    return (
                        "\n---\n".join(results)
                        if results
                        else "No web search results found."
                    )
        raise ValueError("HTTP redirect limit exceeded.")
    finally:
        _network_slots.release()
