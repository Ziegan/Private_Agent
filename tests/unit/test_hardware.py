from unittest.mock import patch, MagicMock
from rich.console import Console

from private_agent.tools import (
    _PublicOnlySyncHTTPTransport,
    _PublicOnlyAsyncHTTPTransport,
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

from private_agent.rag import (
    initialize_knowledge_base,
)
from private_agent.hardware import (
    format_hardware_status,
    inspect_ollama_hardware,
    normalize_acceleration_mode,
    ollama_acceleration_options,
)

console = Console()

def test_hardware_acceleration_policy_options():
    assert normalize_acceleration_mode("auto") == "auto"
    assert normalize_acceleration_mode("CPU") == "cpu"
    assert normalize_acceleration_mode("prefer-accelerator") == "accelerator"
    assert normalize_acceleration_mode("unsupported") == "auto"
    assert ollama_acceleration_options("auto") == {}
    assert ollama_acceleration_options("accelerator") == {}
    assert ollama_acceleration_options("cpu") == {"num_gpu": 0}

def test_local_chat_model_applies_configured_acceleration_mode():
    from private_agent.agent import runtime as agent

    for mode, expected in (
        ("auto", {}),
        ("accelerator", {}),
        ("cpu", {"num_gpu": 0}),
    ):
        with patch("private_agent.agent.runtime.HARDWARE_ACCELERATION_MODE", mode), patch(
            "private_agent.agent.runtime.ChatOllama"
        ) as chat_model:
            agent._make_chat_model(
                "test-model",
                thinking_enabled=False,
                thinking_effort="medium",
                supports_thinking=False,
            )
        actual_options = chat_model.call_args.kwargs
        if expected:
            assert actual_options["num_gpu"] == expected["num_gpu"]
        else:
            assert "num_gpu" not in actual_options
        assert actual_options["sync_client_kwargs"]["transport"].__class__ is (
            _PublicOnlySyncHTTPTransport
        )
        assert actual_options["async_client_kwargs"]["transport"].__class__ is (
            _PublicOnlyAsyncHTTPTransport
        )

    with patch("private_agent.agent.runtime.HARDWARE_ACCELERATION_MODE", "auto"), patch(
        "private_agent.agent.runtime.ChatOllama"
    ) as chat_model:
        agent._make_chat_model(
            "test-model",
            thinking_enabled=False,
            thinking_effort="high",
            supports_thinking=True,
        )
    assert chat_model.call_args.kwargs["reasoning"] is False
    assert "think" not in chat_model.call_args.kwargs

def test_hardware_status_reports_ollama_vram_and_server_host():
    class FakeOllama:
        def ps(self):
            return {
                "models": [
                    {
                        "name": "local-model",
                        "size": 8_000_000_000,
                        "size_vram": 5 * 1024**3,
                    }
                ]
            }

    status = inspect_ollama_hardware(
        "http://localhost:11434",
        client_factory=lambda **kwargs: FakeOllama(),
    )
    assert status["server"] == "local Ollama server (localhost)"
    assert status["placement"] == "GPU VRAM allocation reported by Ollama"
    assert status["models"][0]["size_vram"] == 5 * 1024**3
    assert "does not identify CUDA, ROCm, Vulkan" in status["detail"]
    assert "5.00 GiB VRAM" in format_hardware_status(status)

def test_hardware_status_remote_server_and_cpu_fallback_reporting():
    class FakeOllama:
        def ps(self):
            return {"models": [{"model": "remote-model", "size_vram": 0}]}

    status = inspect_ollama_hardware(
        "http://192.0.2.10:11434",
        "cpu",
        client_factory=lambda **kwargs: FakeOllama(),
    )
    assert "192.0.2.10" in status["server"]
    assert "not the CLI machine" in status["server"]
    assert status["placement"] == "Ollama reports no VRAM allocation for loaded models"
    formatted = format_hardware_status(status)
    assert "CPU requested (Ollama num_gpu=0)" in formatted
    assert "no VRAM allocation" in formatted

def test_hardware_status_handles_unavailable_ollama_server():
    def unavailable(**kwargs):
        raise ConnectionError("connection refused")

    status = inspect_ollama_hardware(
        "http://localhost:11434",
        client_factory=unavailable,
    )
    assert status["placement"] == "unknown"
    assert "Could not query Ollama runtime status" in status["error"]

def test_rag_embeddings_receive_cpu_acceleration_policy(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("Local hardware policy test.", encoding="utf-8")
    monkeypatch.setattr(
        "private_agent.rag.indexing.HARDWARE_ACCELERATION_MODE", "cpu"
    )
    with patch("private_agent.rag.indexing.OllamaEmbeddings") as embeddings, patch(
        "private_agent.rag.indexing.Chroma"
    ) as chroma:
        chroma.from_documents.return_value = MagicMock()
        assert initialize_knowledge_base(str(docs)) is not None
    assert embeddings.call_args.kwargs["num_gpu"] == 0
