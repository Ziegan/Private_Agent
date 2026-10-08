from private_agent.agent.providers import create_local_chat_model


def test_local_model_receives_bounded_context_and_output_options():
    captured = {}

    def model_factory(**options):
        captured.update(options)
        return object()

    create_local_chat_model(
        "local-model",
        temperature=0.1,
        base_url="http://localhost:11434",
        acceleration_mode="auto",
        thinking=None,
        chat_model_factory=model_factory,
        client_kwargs=lambda: {},
        acceleration_options=lambda _mode: {},
        track_clients=lambda _model: None,
        max_output_tokens=512,
        context_window=4096,
    )

    assert captured["num_ctx"] == 4096
    assert captured["num_predict"] == 512


def test_openai_compatible_model_receives_injected_policy_clients():
    from private_agent.agent.providers import create_openai_compatible_chat_model

    captured = {}
    sync_client = object()
    async_client = object()

    def model_factory(**options):
        captured.update(options)
        return "model"

    def sync_client_factory(**options):
        captured["sync_client_options"] = options
        return sync_client

    def async_client_factory(**options):
        captured["async_client_options"] = options
        return async_client

    model = create_openai_compatible_chat_model(
        "https://api.example.test/v1",
        "test-model",
        "session-key",
        temperature=0.2,
        request_timeout=45,
        max_retries=2,
        max_tokens=1024,
        chat_model_factory=model_factory,
        sync_client_factory=sync_client_factory,
        async_client_factory=async_client_factory,
    )

    assert model == "model"
    assert captured["model"] == "test-model"
    assert captured["base_url"] == "https://api.example.test/v1"
    assert captured["api_key"] == "session-key"
    assert captured["temperature"] == 0.2
    assert captured["timeout"] == 45
    assert captured["max_retries"] == 2
    assert captured["max_tokens"] == 1024
    assert captured["http_client"] is sync_client
    assert captured["http_async_client"] is async_client
    assert captured["sync_client_options"] == {
        "headers": {},
        "timeout": 45,
        "allow_loopback": True,
    }
    assert captured["async_client_options"] == {
        "headers": {},
        "timeout": 45,
        "allow_loopback": True,
        "track": True,
    }


def test_online_capabilities_use_model_profile_without_guessing():
    from types import SimpleNamespace

    from private_agent.agent.providers import online_capabilities

    known = SimpleNamespace(
        profile={
            "tool_calling": True,
            "structured_output": True,
            "image_inputs": True,
            "audio_inputs": False,
            "reasoning_output": False,
        }
    )
    assert online_capabilities(known, True) == {
        "tools": True,
        "function_calls": True,
        "structured_output": True,
        "thinking": False,
        "vision": True,
        "audio": False,
    }
    disabled = online_capabilities(known, False)
    assert disabled["tools"] is False and disabled["function_calls"] is False
    unknown = online_capabilities(SimpleNamespace(profile={}), True)
    assert all(value is None for value in unknown.values())
    assert online_capabilities(object(), True)["vision"] is None


def test_local_runtime_discovery_reports_only_reachable_loopback_runtimes():
    from types import SimpleNamespace

    from private_agent.agent.local_runtimes import discover_local_runtimes

    responses = {
        "http://127.0.0.1:11434/api/tags": {
            "models": [
                {"name": "qwen:latest"},
                {"name": "nomic-embed-text"},
            ]
        },
        "http://127.0.0.1:1234/v1/models": {
            "data": [{"id": "lmstudio-model"}]
        },
        "http://127.0.0.1:8080/v1/models": {"data": [{"id": "gguf-model"}]},
    }

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: responses[url],
            )

    runtimes = discover_local_runtimes(
        "http://127.0.0.1:11434",
        [
            {"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"},
            {"name": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1"},
            {"name": "remote server", "base_url": "https://models.example.test/v1"},
        ],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("Ollama", ("qwen:latest",)),
        ("LM Studio", ("lmstudio-model",)),
        ("llama.cpp", ("gguf-model",)),
    ]


def test_local_runtime_discovery_accepts_reachable_server_with_no_models():
    from types import SimpleNamespace

    from private_agent.agent.local_runtimes import discover_local_runtimes

    class FakeClient:
        is_closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": []},
            )

    runtimes = discover_local_runtimes(
        "http://localhost:11434",
        [],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert len(runtimes) == 1
    assert runtimes[0].name == "Ollama"
    assert runtimes[0].models == ()


def test_local_runtime_discovery_survives_and_traces_client_close_failure(
    monkeypatch,
):
    from types import SimpleNamespace

    import private_agent.agent.local_runtimes as local_runtimes
    from private_agent.agent.local_runtimes import discover_local_runtimes

    events = []
    monkeypatch.setattr(
        local_runtimes,
        "log_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    class FakeClient:
        is_closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"models": [{"name": "local-chat"}]},
            )

        def close(self):
            raise OSError("close failed")

    runtimes = discover_local_runtimes(
        "http://localhost:11434",
        [],
        client_factory=lambda **_kwargs: FakeClient(),
    )

    assert [(runtime.name, runtime.models) for runtime in runtimes] == [
        ("Ollama", ("local-chat",))
    ]
    assert events == [
        (
            "provider.local_discovery_client_close_failed",
            {
                "level": local_runtimes.logging.WARNING,
                "runtime_name": "Ollama",
                "error_type": "OSError",
                "exc_info": True,
            },
        )
    ]


def test_local_runtime_discovery_rejects_non_loopback_urls():
    from private_agent.agent.local_runtimes import discover_local_runtimes

    def no_request(**_kwargs):
        raise AssertionError("Non-loopback endpoints must never be contacted.")

    assert discover_local_runtimes(
        "https://remote-ollama.example.test",
        [{"name": "custom", "base_url": "http://192.168.1.10:1234/v1"}],
        client_factory=no_request,
    ) == []


def test_windows_discovers_configured_local_runtimes(monkeypatch):
    from types import SimpleNamespace

    import private_agent.agent.runtime as agent

    discovered = [
        agent.LocalRuntime(
            "LM Studio",
            "openai-compatible",
            "http://127.0.0.1:1234/v1",
            ("local-model",),
        )
    ]
    calls = []

    def discover(*args, **kwargs):
        calls.append((args, kwargs))
        return discovered

    monkeypatch.setattr(agent, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(agent, "discover_local_runtimes", discover)

    assert agent._discover_available_local_runtimes() == discovered
    assert len(calls) == 1
