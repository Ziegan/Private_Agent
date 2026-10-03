"""Hardware policy and runtime status for Ollama-backed inference."""

from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import ollama


ACCELERATION_MODES = frozenset({"auto", "cpu", "accelerator"})


def normalize_acceleration_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "prefer_accelerator": "accelerator",
        "gpu": "accelerator",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in ACCELERATION_MODES else "auto"


def ollama_acceleration_options(mode: str) -> dict[str, int]:
    """Return portable Ollama request options for the configured policy."""
    normalized = normalize_acceleration_mode(mode)
    if normalized == "cpu":
        return {"num_gpu": 0}
    return {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _server_label(base_url: str) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if not host:
        return "host could not be identified"
    if host.lower() in {"localhost", "127.0.0.1", "::1"}:
        return f"local Ollama server ({host})"
    return f"configured Ollama server ({host}; not the CLI machine unless configured that way)"


def inspect_ollama_hardware(
    base_url: str,
    mode: str = "auto",
    *,
    client_factory: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Inspect loaded-model placement reported by the configured Ollama server."""
    normalized_mode = normalize_acceleration_mode(mode)
    result: dict[str, Any] = {
        "server": _server_label(base_url),
        "mode": normalized_mode,
        "models": [],
        "placement": "unknown",
        "detail": (
            "Ollama selects the hardware backend. Its process-list API reports VRAM "
            "allocation, but does not identify CUDA, ROCm, Vulkan, Metal, or OpenVINO."
        ),
        "error": None,
    }
    factory = client_factory or ollama.Client
    client = None
    try:
        client = factory(host=base_url)
        response = client.ps()
    except Exception as exc:
        result["error"] = f"Could not query Ollama runtime status ({type(exc).__name__})."
        return result
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                result["error"] = (
                    f"Could not close Ollama runtime client ({type(exc).__name__})."
                )

    models = _field(response, "models", []) or []
    for model in models:
        name = _field(model, "name") or _field(model, "model") or "unnamed model"
        vram = _field(model, "size_vram")
        if isinstance(vram, int) and vram >= 0:
            result["models"].append({"name": str(name), "size_vram": vram})
        else:
            result["models"].append({"name": str(name), "size_vram": None})

    if not models:
        result["placement"] = "no loaded models; active placement is unknown"
    elif any(
        isinstance(_field(model, "size_vram"), int)
        and _field(model, "size_vram") > 0
        for model in models
    ):
        result["placement"] = "GPU VRAM allocation reported by Ollama"
    elif all(_field(model, "size_vram") == 0 for model in models):
        result["placement"] = "Ollama reports no VRAM allocation for loaded models"
    else:
        result["placement"] = "loaded-model hardware placement is unknown"
    return result


def format_hardware_status(status: dict[str, Any]) -> str:
    mode = status["mode"]
    if mode == "auto":
        policy = "automatic; Ollama chooses the available runtime"
    elif mode == "cpu":
        policy = "CPU requested (Ollama num_gpu=0); server support may vary"
    else:
        policy = (
            "accelerator preferred; Ollama's request API cannot enforce this, "
            "so its default selection is used"
        )

    lines = [
        f"Server: {status['server']}",
        f"Policy: {policy}",
        f"Loaded model placement: {status['placement']}",
    ]
    if status["models"]:
        model_details = []
        for model in status["models"]:
            allocation = model["size_vram"]
            allocation_text = (
                "VRAM unknown"
                if allocation is None
                else f"{allocation / (1024 ** 3):.2f} GiB VRAM"
            )
            model_details.append(f"{model['name']} ({allocation_text})")
        lines.append("Loaded models: " + ", ".join(model_details))
    if status["error"]:
        lines.append(status["error"])
    lines.append(status["detail"])
    return "\n".join(lines)
