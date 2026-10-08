import pytest
import asyncio
import sys
import types
from unittest.mock import patch, MagicMock, AsyncMock
from rich.console import Console

from private_agent.tools import (
    download_web_file,
    load_configured_mcp_tools,
    check_internet_connection,
    validate_outbound_url,
    _get_limited_response,
    _PublicOnlyNetworkBackend,
    _PublicOnlySyncHTTPTransport,
    _PublicOnlyAsyncHTTPTransport,
    close_outbound_http_clients,
    _search_duckduckgo,
    _validate_explicit_service_url,
)
from private_agent.tools import network as network_tools
from private_agent.agent import (
    _online_request_failure,
    execute_tool_call,
    _validate_online_base_url,
    _make_online_chat_model,
    select_online_model,
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


@pytest.fixture(autouse=True)
def sync_mocked_resolver_with_async_backend(monkeypatch):
    original_sync = network_tools._getaddrinfo_with_timeout
    original_async = network_tools._getaddrinfo_async_with_timeout

    async def resolve_async(host, port, timeout):
        if network_tools._getaddrinfo_with_timeout is original_sync:
            return await original_async(host, port, timeout)
        return network_tools._getaddrinfo_with_timeout(host, port, timeout)

    monkeypatch.setattr(
        network_tools,
        "_getaddrinfo_async_with_timeout",
        resolve_async,
    )


def test_internet_probe_uses_per_call_timeout_without_global_socket_mutation():
    with patch("private_agent.tools.socket.create_connection") as connect:
        assert check_internet_connection("example.test", 443, 0.25) is True
        connect.assert_called_once()
        assert connect.call_args.kwargs["timeout"] == 0.25


def test_outbound_client_close_failure_is_logged(monkeypatch):
    events = []
    client = MagicMock()
    client.close.side_effect = OSError("close failed")
    monkeypatch.setattr(network_tools, "_outbound_http_clients", [client])
    monkeypatch.setattr(
        network_tools,
        "log_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    asyncio.run(close_outbound_http_clients())

    assert events == [
        (
            "network.client_close_failed",
            {
                "level": network_tools.logging.WARNING,
                "client_type": "MagicMock",
                "error_type": "OSError",
            },
        )
    ]


def test_outbound_url_rejects_local_and_credentialed_destinations():
    for url in (
        "http://127.0.0.1/admin",
        "http://localhost/private",
        "file:///etc/passwd",
        "http://user:pass@example.com/",
    ):
        with pytest.raises(ValueError):
            validate_outbound_url(url)

def test_outbound_url_rejects_domains_resolving_to_private_addresses(monkeypatch):
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("10.0.0.12", 80)),
        ],
    )
    with pytest.raises(ValueError, match="non-public"):
        validate_outbound_url("http://internal.example.test/")


def test_resolver_child_timeout_is_bounded_and_reported(monkeypatch):
    import subprocess
    import httpcore

    run = MagicMock(side_effect=subprocess.TimeoutExpired("resolver", 0.025))
    monkeypatch.setattr(network_tools.subprocess, "run", run)

    with pytest.raises(httpcore.ConnectTimeout, match="DNS resolution timed out"):
        network_tools._getaddrinfo_with_timeout("slow.example.test", 443, 0.025)

    assert run.call_args.kwargs["timeout"] == 0.025
    assert run.call_args.kwargs["check"] is True


def test_stalled_system_dns_child_is_terminated_within_deadline(monkeypatch):
    import httpcore
    import time

    monkeypatch.setattr(
        network_tools,
        "_RESOLVER_SCRIPT",
        "import time; time.sleep(5)",
    )
    started = time.monotonic()
    with pytest.raises(httpcore.ConnectTimeout, match="DNS resolution timed out"):
        network_tools._getaddrinfo_with_timeout("stalled.example.test", 443, 0.1)

    assert time.monotonic() - started < 1


@pytest.mark.asyncio
async def test_async_resolver_kills_and_reaps_child_on_timeout(monkeypatch):
    import httpcore

    class DelayedResolver:
        returncode = None

        def __init__(self):
            self.killed = False
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            self.returncode = -9
            return b"", b""

        def kill(self):
            self.killed = True
            self.finished.set()

    process = DelayedResolver()
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(
        network_tools.asyncio,
        "create_subprocess_exec",
        create_process,
    )

    with pytest.raises(httpcore.ConnectTimeout, match="DNS resolution timed out"):
        await network_tools._getaddrinfo_async_with_timeout(
            "slow.example.test", 443, 0.01
        )

    assert process.killed
    create_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_stalled_async_dns_child_is_terminated_within_deadline(monkeypatch):
    import httpcore
    import time

    monkeypatch.setattr(
        network_tools,
        "_RESOLVER_SCRIPT",
        "import time; time.sleep(5)",
    )
    started = time.monotonic()
    with pytest.raises(httpcore.ConnectTimeout, match="DNS resolution timed out"):
        await network_tools._getaddrinfo_async_with_timeout(
            "stalled.example.test", 443, 0.1
        )

    assert time.monotonic() - started < 1


@pytest.mark.asyncio
async def test_async_resolver_kills_and_reaps_child_on_cancellation(monkeypatch):
    class DelayedResolver:
        returncode = None

        def __init__(self):
            self.killed = False
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            self.returncode = -9
            return b"", b""

        def kill(self):
            self.killed = True
            self.finished.set()

    process = DelayedResolver()
    monkeypatch.setattr(
        network_tools.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    task = asyncio.create_task(
        network_tools._getaddrinfo_async_with_timeout(
            "slow.example.test", 443, 5
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed

@pytest.mark.asyncio
async def test_connect_backend_pins_validated_public_address(monkeypatch):
    backend = _PublicOnlyNetworkBackend()
    connect = AsyncMock(return_value="connected")
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("1.1.1.1", 443)),
            (2, 1, 6, "", ("8.8.8.8", 443)),
        ],
    )
    result = await backend.connect_tcp("public.example.test", 443)
    assert result == "connected"
    assert connect.await_args.args[0] == "1.1.1.1"


def test_sync_connect_timeout_budget_includes_dns(monkeypatch):
    import time

    from private_agent.tools.network import _PublicOnlySyncNetworkBackend

    backend = _PublicOnlySyncNetworkBackend()
    observed = {}

    def resolve(host, port, timeout):
        time.sleep(0.02)
        return ["1.1.1.1"]

    def connect(address, port, timeout=None, **kwargs):
        observed["timeout"] = timeout
        return "connected"

    monkeypatch.setattr(network_tools, "_getaddrinfo_with_timeout", resolve)
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)

    assert backend.connect_tcp("public.example.test", 443, timeout=0.2) == "connected"
    assert 0 < observed["timeout"] < 0.19


@pytest.mark.asyncio
async def test_async_connect_timeout_budget_includes_dns(monkeypatch):
    backend = _PublicOnlyNetworkBackend()
    observed = {}

    async def resolve(host, port, timeout):
        await asyncio.sleep(0.02)
        return ["1.1.1.1"]

    async def connect(address, port, timeout=None, **kwargs):
        observed["timeout"] = timeout
        return "connected"

    monkeypatch.setattr(network_tools, "_getaddrinfo_async_with_timeout", resolve)
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)

    assert await backend.connect_tcp(
        "public.example.test", 443, timeout=0.2
    ) == "connected"
    assert 0 < observed["timeout"] < 0.19


@pytest.mark.asyncio
async def test_connect_backend_falls_back_from_unavailable_localhost_ipv6(monkeypatch):
    backend = _PublicOnlyNetworkBackend(allow_loopback=True)
    attempts = []

    async def connect(address, port, **kwargs):
        attempts.append(address)
        if address == "::1":
            import httpcore
            raise httpcore.ConnectError("IPv6 listener unavailable")
        return "connected"

    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [
            (10, 1, 6, "", ("::1", 11434, 0, 0)),
            (2, 1, 6, "", ("127.0.0.1", 11434)),
        ],
    )
    assert await backend.connect_tcp("localhost", 11434) == "connected"
    assert attempts == ["::1", "127.0.0.1"]

@pytest.mark.asyncio
async def test_connect_backend_rejects_mixed_public_private_dns(monkeypatch):
    backend = _PublicOnlyNetworkBackend()
    connect = AsyncMock()
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [
            (2, 1, 6, "", ("1.1.1.1", 443)),
            (2, 1, 6, "", ("10.0.0.5", 443)),
        ],
    )
    with pytest.raises(OSError, match="non-public"):
        await backend.connect_tcp("mixed.example.test", 443)
    connect.assert_not_awaited()

@pytest.mark.asyncio
async def test_private_provider_transport_allows_only_explicit_private_literals(monkeypatch):
    backend = _PublicOnlyNetworkBackend(allow_loopback=True)
    connect = AsyncMock(return_value="connected")
    monkeypatch.setattr(backend._backend, "connect_tcp", connect)
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda host, port, *args: [(2, 1, 6, "", (host, port))],
    )
    assert await backend.connect_tcp("192.168.1.20", 8080) == "connected"
    connect.assert_awaited_once()

    for hostname, resolved in (
        ("provider.example.test", "192.168.1.20"),
        ("169.254.169.254", "169.254.169.254"),
    ):
        monkeypatch.setattr(
            "private_agent.tools.network._getaddrinfo_with_timeout",
            lambda *args, address=resolved, **kwargs: [
                (2, 1, 6, "", (address, 443))
            ],
        )
        with pytest.raises(OSError, match="non-public"):
            await backend.connect_tcp(hostname, 443)


@pytest.mark.asyncio
@pytest.mark.parametrize("private_address", ["127.0.0.1", "::1"])
async def test_real_http_transport_blocks_private_dns_results(
    monkeypatch, private_address
):
    import httpx

    async def respond(reader, writer):
        request_received.set()
        await reader.read(4096)
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
            b"Connection: close\r\n\r\nok"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    request_received = asyncio.Event()
    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [
            (
                10 if ":" in private_address else 2,
                1,
                6,
                "",
                (private_address, port, 0, 0)
                if ":" in private_address
                else (private_address, port),
            )
        ],
    )

    try:
        async with httpx.AsyncClient(
            transport=_PublicOnlyAsyncHTTPTransport(),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            with pytest.raises(OSError, match="non-public"):
                await client.get(f"http://public.example.test:{port}/")
        assert not request_received.is_set()
    finally:
        server.close()
        await server.wait_closed()


def test_real_sync_http_transport_blocks_private_dns_results(monkeypatch):
    import httpx
    import socket
    import threading

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(1)
    port = listener.getsockname()[1]
    connected = threading.Event()

    def accept_request():
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        connected.set()
        connection.close()

    server_thread = threading.Thread(target=accept_request)
    server_thread.start()
    monkeypatch.setattr(
        network_tools,
        "_getaddrinfo_with_timeout",
        lambda host, port, timeout: ["127.0.0.1"],
    )

    try:
        with httpx.Client(
            transport=_PublicOnlySyncHTTPTransport(),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            with pytest.raises(OSError, match="non-public"):
                client.get(f"http://public.example.test:{port}/")
        assert not connected.is_set()
    finally:
        listener.close()
        server_thread.join(timeout=2)


@pytest.mark.asyncio
async def test_http_redirect_to_local_network_is_rejected(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"Location": "http://127.0.0.1/admin"},
        )
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(ValueError, match="non-public"):
            await _get_limited_response(client, "http://public.example.test/", 100)

@pytest.mark.asyncio
async def test_http_response_size_limit_is_enforced(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 80))],
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"too large")
    )
    async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
        with pytest.raises(ValueError, match="size limit"):
            await _get_limited_response(client, "http://public.example.test/", 4)

@pytest.mark.asyncio
async def test_http_total_timeout_releases_concurrency_slot(monkeypatch):
    import private_agent.tools as tools

    class SlowClient:
        def stream(self, *args, **kwargs):
            class SlowResponse:
                async def __aenter__(self):
                    await asyncio.sleep(0.1)

                async def __aexit__(self, *args):
                    pass
            return SlowResponse()

    monkeypatch.setattr("private_agent.tools.network.NETWORK_REQUEST_TIMEOUT", 0.01)
    async def validate_url(url):
        return url

    monkeypatch.setattr(
        "private_agent.tools.network._validate_outbound_url_async",
        validate_url,
    )
    with pytest.raises(asyncio.TimeoutError):
        await _get_limited_response(SlowClient(), "https://public.test/", 100)
    assert tools._network_slots.acquire(blocking=False)
    tools._network_slots.release()


@pytest.mark.asyncio
async def test_async_request_timeout_cancels_dns_validation(monkeypatch):
    import httpcore
    import private_agent.tools as tools

    cancelled = asyncio.Event()

    async def stalled_resolver(host, port, timeout):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(
        "private_agent.tools.network.NETWORK_REQUEST_TIMEOUT",
        0.01,
    )
    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_async_with_timeout",
        stalled_resolver,
    )
    with pytest.raises((asyncio.TimeoutError, httpcore.ConnectTimeout)):
        await _get_limited_response(
            MagicMock(),
            "https://stalled.example.test/",
            100,
        )

    assert cancelled.is_set()
    assert tools._network_slots.acquire(blocking=False)
    tools._network_slots.release()


def test_failed_download_preserves_existing_destination(temp_workspace, monkeypatch):
    destination = temp_workspace / "existing.txt"
    destination.write_text("original", encoding="utf-8")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("private_agent.tools.web.has_internet_connection", lambda: True)
    monkeypatch.setattr(
        "private_agent.tools.web._public_only_async_client",
        lambda **kwargs: FakeClient(),
    )

    async def fail_fetch(*args, **kwargs):
        raise OSError("network dropped")

    monkeypatch.setattr("private_agent.tools.web._get_limited_response", fail_fetch)
    result = download_web_file.invoke({
        "url": "https://public.example.test/file.txt",
        "save_path": "existing.txt",
    })
    assert "network dropped" in result
    assert destination.read_text(encoding="utf-8") == "original"
    assert list(temp_workspace.glob(".existing.txt.*.part")) == []

def test_online_api_url_normalizes_v1_and_requires_secure_remote_transport():
    assert _validate_online_base_url("https://api.example.test") == (
        "https://api.example.test/v1"
    )
    assert _validate_online_base_url("http://localhost:8080/v1/") == (
        "http://localhost:8080/v1"
    )
    with pytest.raises(ValueError):
        _validate_online_base_url("http://api.example.test/v1")
    with pytest.raises(ValueError):
        _validate_online_base_url("https://user:secret@api.example.test/v1")

@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, "provider rejected the request"),
        (401, "rejected authentication"),
        (403, "denied access"),
        (404, "not found"),
        (429, "rate limit or account quota"),
        (503, "server error"),
    ],
)
def test_online_request_failure_reports_status_without_response_secrets(
    status_code, expected
):
    failure = types.SimpleNamespace(
        status_code=status_code,
        response=types.SimpleNamespace(
            status_code=status_code,
            text="sensitive-provider-response",
        ),
    )

    message = _online_request_failure(failure)

    assert f"HTTP {status_code}" in message
    assert expected in message
    assert "sensitive-provider-response" not in message

def test_online_request_failure_without_status_hides_exception_details():
    failure = RuntimeError("secret-token and private response body")

    message = _online_request_failure(failure)

    assert "check connectivity, endpoint, model, and credentials" in message
    assert "secret-token" not in message
    assert "private response body" not in message

def test_online_model_uses_openai_compatible_client_with_session_key():
    adapter = types.ModuleType("langchain_openai")
    adapter.ChatOpenAI = MagicMock(return_value=object())
    with patch.dict(sys.modules, {"langchain_openai": adapter}):
        model = _make_online_chat_model(
            "https://api.example.test/v1", "test-model", "session-secret"
        )
    assert model is not None
    adapter.ChatOpenAI.assert_called_once()
    assert adapter.ChatOpenAI.call_args.kwargs["base_url"] == "https://api.example.test/v1"
    assert adapter.ChatOpenAI.call_args.kwargs["api_key"] == "session-secret"
    assert adapter.ChatOpenAI.call_args.kwargs["http_client"] is not None
    assert adapter.ChatOpenAI.call_args.kwargs["http_async_client"] is not None
    asyncio.run(close_outbound_http_clients())

def test_online_model_listing_uses_safe_transport_and_session_key(monkeypatch):
    import private_agent.agent.runtime as agent

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "model-a"}]}

    class FakeHttpClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            self.url = url
            return FakeResponse()

    client = FakeHttpClient()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(agent, "public_only_sync_client", factory)
    monkeypatch.setattr(agent, "check_internet_connection", lambda *args: True)
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent, "APP_CONFIG", {"online_base_url": "https://api.example.test/v1"})
    monkeypatch.setattr(agent, "getpass", lambda prompt: "ephemeral-test-key")
    monkeypatch.setattr(agent, "_make_online_chat_model", lambda *args: "online-model")
    monkeypatch.setattr(
        agent.console,
        "input",
        MagicMock(side_effect=["", "y", "", "n", "n"]),
    )
    selected = select_online_model()
    assert selected["model"] == "online-model"
    assert selected["model_name"] == "model-a"
    assert "api_key" not in selected
    assert factory.call_args.kwargs["allow_loopback"] is True
    assert client.url == "https://api.example.test/v1/models"
    assert "ephemeral-test-key" not in repr(selected)

def test_online_model_authentication_failure_is_redacted(monkeypatch):
    import httpx
    import private_agent.agent.runtime as agent

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "bad key secret-should-not-leak",
                request=httpx.Request("GET", "https://api.example.test/v1/models"),
                response=httpx.Response(401),
            )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(agent, "public_only_sync_client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(agent, "check_internet_connection", lambda *args: True)
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent, "APP_CONFIG", {"online_base_url": "https://api.example.test/v1"})
    monkeypatch.setattr(agent, "getpass", lambda prompt: "secret-should-not-leak")
    prompt = MagicMock(side_effect=["", "y"])
    monkeypatch.setattr(agent.console, "input", prompt)
    output = []
    monkeypatch.setattr(agent.console, "print", lambda *args, **kwargs: output.append(str(args)))

    assert select_online_model() is None
    assert all("secret-should-not-leak" not in line for line in output)
    assert prompt.call_count == 2

@pytest.mark.asyncio
async def test_network_tool_call_requires_runtime_authorization():
    with patch("private_agent.agent.runtime.ENABLE_WEB_RESEARCH", True), \
         patch("private_agent.agent.runtime.WEB_RESEARCH_CONSENT", "ask"), \
         patch("private_agent.agent.runtime.AVAILABLE_TOOLS", {"web_search": MagicMock()}) as tools:
        result = await execute_tool_call(
            {"name": "web_search", "args": {"query": "private query"}, "id": "call-1"}
        )
    assert "not authorized" in result.content
    tools["web_search"].invoke.assert_not_called()

def test_explicit_mcp_http_endpoints_are_peer_validated(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "private_agent.tools.network._resolve_validated_peer",
        lambda host, port, allow_loopback: checked.append(
            (host, port, allow_loopback)
        ) or [__import__("ipaddress").ip_address("203.0.113.5")],
    )
    assert _validate_explicit_service_url(
        "http://mcp.example.test:8080/sse", {"http", "https"}
    ) == "http://mcp.example.test:8080/sse"
    assert checked == [("mcp.example.test", 8080, True)]
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        _validate_explicit_service_url("wss://mcp.example.test/ws", {"ws", "wss"})

def test_mcp_http_transport_uses_pinned_client_factory(monkeypatch):
    captured = {}

    class FakeMCPClient:
        def __init__(self, configs):
            captured["configs"] = configs

        async def get_tools(self, server_name):
            return []

    parent = types.ModuleType("langchain_mcp_adapters")
    parent.__path__ = []
    adapter = types.ModuleType("langchain_mcp_adapters.client")
    adapter.MultiServerMCPClient = FakeMCPClient
    monkeypatch.setattr(
        "private_agent.tools.network._resolve_validated_peer",
        lambda host, port, allow_loopback: [
            __import__("ipaddress").ip_address("203.0.113.5")
        ],
    )
    client = None
    with patch.dict(
        sys.modules,
        {
            "langchain_mcp_adapters": parent,
            "langchain_mcp_adapters.client": adapter,
        },
    ):
        assert asyncio.run(
            load_configured_mcp_tools(
                "remote",
                {
                    "transport": "streamable_http",
                    "url": "https://mcp.example.test/mcp",
                },
            )
        ) == []
        factory = captured["configs"]["remote"]["httpx_client_factory"]
        client = factory(headers={}, timeout=None, auth=None)
    try:
        assert client.trust_env is False
        assert client.follow_redirects is False
        assert client._transport._pool._network_backend._allow_loopback is True
    finally:
        asyncio.run(client.aclose())

def test_explicit_endpoint_blocks_private_hostname_resolution(monkeypatch):
    import private_agent.tools as tools

    monkeypatch.setattr(
        "private_agent.tools.network._getaddrinfo_with_timeout",
        lambda host, port, *_args: [
            "10.0.0.7"
        ],
    )
    with pytest.raises(OSError, match="non-public address"):
        tools._resolve_validated_peer(
            "mcp.example.test", 8080, allow_loopback=True
        )

def test_duckduckgo_search_uses_bounded_pinned_http_client(monkeypatch):
    html = b"""
    <div class="result">
      <a class="result__a" href="https://docs.example.test">Docs</a>
      <a class="result__snippet">A useful result</a>
    </div>
    """

    class FakeResponse:
        is_redirect = False
        headers = {"content-length": str(len(html))}
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield html

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream(self, method, url, params=None):
            assert method == "GET"
            assert params == {"q": "private agent"}
            return FakeResponse()

    monkeypatch.setattr(
        "private_agent.tools.network.validate_outbound_url", lambda url: url
    )
    result = _search_duckduckgo(
        "private agent",
        client_factory=lambda **kwargs: FakeClient(),
    )
    assert "Title: Docs" in result
    assert "Snippet: A useful result" in result
    assert "https://docs.example.test" in result


def _configured_online_setup(monkeypatch, answers):
    import private_agent.agent.runtime as agent

    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "model-a"}]}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            seen["url"] = url
            return FakeResponse()

    def factory(**kwargs):
        seen["headers"] = kwargs["headers"]
        return FakeClient()

    monkeypatch.setattr(agent, "public_only_sync_client", factory)
    monkeypatch.setattr(agent, "check_internet_connection", lambda *args: True)
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(
        agent,
        "APP_CONFIG",
        {"online_base_url": "https://saved.example.test/v1", "online_api_key": "saved-key"},
    )
    monkeypatch.setattr(agent, "getpass", lambda prompt: "typed-key")
    monkeypatch.setattr(agent, "_make_online_chat_model", lambda *args: "online-model")
    monkeypatch.setattr(agent.console, "input", MagicMock(side_effect=answers))
    return agent, seen


def test_online_selection_uses_saved_url_and_key_when_accepted(monkeypatch):
    agent, seen = _configured_online_setup(monkeypatch, ["", "", "n", "n"])
    selected = agent.select_online_model()
    assert selected["model"] == "online-model"
    assert seen["url"] == "https://saved.example.test/v1/models"
    assert seen["headers"]["Authorization"] == "Bearer saved-key"
    assert "saved-key" not in repr(selected)


def test_online_selection_explicitly_uses_saved_details_without_key_prompt(
    monkeypatch,
):
    agent, seen = _configured_online_setup(monkeypatch, ["1", "", "n", "n"])
    key_prompt = MagicMock(side_effect=AssertionError("Saved key should be reused."))
    monkeypatch.setattr(agent, "getpass", key_prompt)

    selected = agent.select_online_model()

    assert selected["model"] == "online-model"
    assert seen["url"] == "https://saved.example.test/v1/models"
    key_prompt.assert_not_called()


def test_online_selection_loads_nested_saved_config_without_config_prompt(
    monkeypatch,
):
    agent, seen = _configured_online_setup(monkeypatch, ["", "n", "n"])
    agent.APP_CONFIG = {
        "models": {
            "online_base_url": "https://nested.example.test/v1",
            "online_api_key": "nested-config-key",
            "online_model": "configured-model",
        }
    }
    key_prompt = MagicMock(
        side_effect=AssertionError("Saved key must be used directly.")
    )
    monkeypatch.setattr(agent, "getpass", key_prompt)
    input_prompt = agent.console.input
    input_prompt.side_effect = ["", "n", "n"]

    selected = agent.select_online_model(True)

    assert selected["model"] == "online-model"
    assert seen["url"] == "https://nested.example.test/v1/models"
    key_prompt.assert_not_called()
    assert len(input_prompt.call_args_list) == 3


def test_online_selection_allows_manual_entry_over_saved_config(monkeypatch):
    agent, seen = _configured_online_setup(
        monkeypatch, ["n", "https://other.example.test/v1", "y", "", "n", "n"]
    )
    selected = agent.select_online_model()
    assert selected["model"] == "online-model"
    assert seen["url"] == "https://other.example.test/v1/models"
    assert seen["headers"]["Authorization"] == "Bearer typed-key"


def test_online_permission_override_can_consent_to_tool_call_limits(monkeypatch):
    from private_agent.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["models"]["online_permission_override"] is False
    # Prompts: share-context n, model id handled by listing "model-a" -> "" ; no tool prompt.
    agent_mod, seen = _configured_online_setup(monkeypatch, ["", "", "n", "y"])
    monkeypatch.setattr(agent_mod, "ONLINE_PERMISSION_OVERRIDE", True)
    selected = agent_mod.select_online_model()
    assert selected["allow_tools"] is True
    assert selected["enforce_tool_call_limits"] is True


def test_online_permission_override_can_decline_tool_call_limits(monkeypatch):
    agent_mod, _seen = _configured_online_setup(monkeypatch, ["", "", "n", "n"])
    monkeypatch.setattr(agent_mod, "ONLINE_PERMISSION_OVERRIDE", True)

    selected = agent_mod.select_online_model()

    assert selected["allow_tools"] is True
    assert selected["enforce_tool_call_limits"] is False
