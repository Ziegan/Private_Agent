"""Provider-independent validation and error formatting for model backends."""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..run_logging import RUN_LOGGER, log_event


def _response_field(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def inspect_local_model_capabilities(
    model_name: str,
    *,
    base_url: str,
    client_factory: Callable[..., Any],
    client_kwargs: Callable[[], dict],
    capability_cache: dict,
    max_cache_entries: int,
    console: Any,
    close_client: Callable[[Any], None],
) -> Dict[str, Optional[bool] | int]:
    """Read Ollama-declared capabilities, with a bounded per-process cache."""
    cache_key = (base_url, model_name)
    cached = capability_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    result: Dict[str, Optional[bool] | int] = {
        "tools": None,
        "function_calls": None,
        "structured_output": None,
        "thinking": None,
        "vision": None,
        "audio": None,
        "context_window": None,
    }
    client = None
    try:
        client = client_factory(host=base_url, **client_kwargs())
        details = client.show(model_name)
    except Exception as exc:
        log_event(
            RUN_LOGGER,
            "provider.local_capabilities_failed",
            level=logging.WARNING,
            model=model_name,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(
            f"[yellow][Model capabilities] Could not inspect {model_name}: {exc}[/yellow]"
        )
        return result
    finally:
        if client is not None:
            close_client(client)

    raw_capabilities = _response_field(details, "capabilities", [])
    if not raw_capabilities:
        model_info = _response_field(details, "modelinfo", {}) or {}
        raw_capabilities = (
            model_info.get("capabilities", [])
            if isinstance(model_info, dict)
            else []
        )
    model_info = _response_field(details, "modelinfo", {}) or {}
    if isinstance(model_info, dict):
        context_lengths = [
            int(value)
            for key, value in model_info.items()
            if str(key).lower().endswith("context_length")
            and isinstance(value, (int, float))
            and value > 0
        ]
        if context_lengths:
            result["context_window"] = min(context_lengths)
    if not raw_capabilities:
        if len(capability_cache) >= max_cache_entries:
            capability_cache.clear()
        capability_cache[cache_key] = result.copy()
        return result

    names = {
        str(item).lower()
        for item in raw_capabilities
        if isinstance(item, (str, int, float))
    }
    result.update(
        tools="tools" in names,
        function_calls="tools" in names,
        structured_output=(
            "structured_output" in names if "structured_output" in names else None
        ),
        thinking="thinking" in names,
        vision="vision" in names,
        audio="audio" in names,
    )
    if len(capability_cache) >= max_cache_entries:
        capability_cache.clear()
    capability_cache[cache_key] = result.copy()
    return result


def create_local_chat_model(
    model_name: str,
    *,
    temperature: float,
    base_url: str,
    acceleration_mode: str,
    thinking: Optional[bool | str],
    chat_model_factory: Callable[..., Any],
    client_kwargs: Callable[[], dict],
    acceleration_options: Callable[[str], dict],
    track_clients: Callable[[Any], None],
    max_output_tokens: int = 2048,
    context_window: Optional[int] = None,
) -> Any:
    options: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "base_url": base_url,
    }
    options.update(client_kwargs())
    options.update(acceleration_options(acceleration_mode))
    options["num_predict"] = max(1, max_output_tokens)
    if context_window is not None:
        options["num_ctx"] = max(1, context_window)
    if thinking is not None:
        options["reasoning"] = thinking
    model = chat_model_factory(**options)
    track_clients(model)
    return model


def create_openai_compatible_chat_model(
    base_url: str,
    model_name: str,
    api_key: str,
    *,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    max_tokens: int,
    chat_model_factory: Callable[..., Any],
    sync_client_factory: Callable[..., Any],
    async_client_factory: Callable[..., Any],
) -> Any:
    """Construct an OpenAI-compatible model with policy-controlled clients."""
    http_client = sync_client_factory(
        headers={},
        timeout=request_timeout,
        allow_loopback=True,
    )
    http_async_client = async_client_factory(
        headers={},
        timeout=request_timeout,
        allow_loopback=True,
        track=True,
    )
    return chat_model_factory(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout=request_timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        http_client=http_client,
        http_async_client=http_async_client,
    )


def create_robust_local_chat_model(
    primary_model_name: str,
    fallback_model_name: str,
    tools: Optional[Sequence[Any]],
    *,
    temperature: float,
    base_url: str,
    acceleration_mode: str,
    thinking: Optional[bool | str],
    chat_model_factory: Callable[..., Any],
    client_kwargs: Callable[[], dict],
    acceleration_options: Callable[[str], dict],
    track_clients: Callable[[Any], None],
    console: Any,
    max_output_tokens: int = 2048,
    context_window: Optional[int] = None,
) -> Any:
    """Create a local model, retrying a compatible fallback when appropriate."""
    last_error: Optional[Exception] = None
    for model_name in dict.fromkeys((primary_model_name, fallback_model_name)):
        try:
            model = create_local_chat_model(
                model_name,
                temperature=temperature,
                base_url=base_url,
                acceleration_mode=acceleration_mode,
                thinking=thinking,
                chat_model_factory=chat_model_factory,
                client_kwargs=client_kwargs,
                acceleration_options=acceleration_options,
                track_clients=track_clients,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
            )
            return model.bind_tools(list(tools)) if tools is not None else model
        except Exception as exc:
            last_error = exc
            log_event(
                RUN_LOGGER,
                "provider.local_model_initialization_failed",
                level=logging.WARNING,
                model=model_name,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            console.print(
                f"[yellow][Model fallback] {model_name} unavailable: {exc}[/yellow]"
            )
            message = str(exc).lower()
            retryable = isinstance(exc, (ConnectionError, TimeoutError)) or any(
                marker in message
                for marker in (
                    "connection refused",
                    "connection error",
                    "timed out",
                    "timeout",
                    "model not found",
                    "status code: 404",
                    "status code: 503",
                    "does not support tools",
                    "tool calling is not supported",
                )
            )
            if model_name == primary_model_name and not retryable:
                break
    if last_error:
        console.print(
            f"[red][Model error] No usable local model: {last_error}[/red]"
        )
    return None


def validate_online_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("Enter a valid HTTP(S) OpenAI-compatible base URL.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("The API base URL contains an invalid port.") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The API base URL must not contain credentials, query, or fragment data.")
    host = parsed.hostname.lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not is_local:
        raise ValueError("Remote API endpoints must use HTTPS.")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def online_request_failure(exc: Exception) -> str:
    """Describe provider HTTP failures without exposing response bodies or credentials."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = None
    if status_code is None:
        return (
            f"Online request failed ({type(exc).__name__}); check connectivity, "
            "endpoint, model, and credentials. Provider response details were not logged."
        )

    explanations = {
        400: "the provider rejected the request; the model or request format may be unsupported",
        401: "the provider rejected authentication; verify the API key",
        403: "the provider denied access to this model or account",
        404: "the endpoint or model was not found",
        408: "the provider timed out processing the request",
        413: "the request exceeded the provider's size limit",
        422: "the provider could not process the request or model parameters",
        429: "the provider rate limit or account quota was reached",
    }
    explanation = explanations.get(
        status_code,
        "the provider returned a server error" if status_code >= 500
        else "the provider rejected the request",
    )
    return (
        f"Online request failed with HTTP {status_code} ({type(exc).__name__}): "
        f"{explanation}. Check the selected model, provider access, and request compatibility. "
        "Response bodies and credentials were not logged."
    )


def has_saved_online_config(app_config: dict) -> bool:
    """Return whether a non-empty API key is available in categorized or legacy config."""
    models = app_config.get("models")
    configured_key = models.get("online_api_key") if isinstance(models, dict) else None
    if configured_key is None:
        configured_key = app_config.get("online_api_key")
    return isinstance(configured_key, str) and bool(configured_key.strip())


def select_online_model(
    *,
    app_config: dict,
    console: Any,
    is_interactive: Callable[[], bool],
    validate_base_url: Callable[[str], str],
    check_connection: Callable[[str, int, float], bool],
    connection_timeout: float,
    get_api_key: Callable[[str], str],
    model_list_timeout: float,
    model_list_limit: int,
    public_sync_client: Callable[..., Any],
    model_factory: Callable[[str, str, str], Any],
    permission_override: bool = False,
    use_saved_details: Optional[bool] = None,
    max_tool_iterations: int,
    max_tool_calls: int,
) -> Optional[dict]:
    """Collect session-only OpenAI-compatible settings after connectivity check."""
    if not is_interactive():
        console.print(
            "[yellow]Online mode requires an interactive terminal for secure key entry.[/yellow]"
        )
        return None

    model_config = app_config.get("models")
    model_config = model_config if isinstance(model_config, dict) else {}
    configured_url = str(
        model_config.get(
            "online_base_url",
            app_config.get("online_base_url", "https://api.openai.com/v1"),
        )
    )
    configured_key_value = model_config.get(
        "online_api_key",
        app_config.get("online_api_key"),
    )
    configured_key = (
        configured_key_value.strip()
        if isinstance(configured_key_value, str)
        else ""
    )
    if use_saved_details is True and not configured_key:
        console.print(
            "[red]Saved online configuration was selected, but no API key is "
            "configured. Choose manual entry or update the config file.[/red]"
        )
        return None
    use_configured = use_saved_details is True
    if use_saved_details is False:
        use_configured = False
    elif use_saved_details is None and configured_key:
        answer = console.input(
            f"[cyan]Online configuration: 1. Use saved details ({configured_url}; "
            "key hidden), 2. Enter URL and key manually [1]: [/cyan]"
        ).strip().lower()
        if answer in {"", "1", "y", "yes"}:
            use_configured = True
        elif answer in {"2", "n", "no"}:
            use_configured = False
        else:
            console.print("[red]Choose 1 for saved details or 2 for manual entry.[/red]")
            return None

    if use_configured:
        try:
            base_url = validate_base_url(configured_url)
        except ValueError as exc:
            console.print(f"[red]Invalid configured endpoint: {exc}[/red]")
            return None
    else:
        base_input = console.input(
            f"[cyan]OpenAI-compatible API base URL [{configured_url}]: [/cyan]"
        ).strip()
        try:
            base_url = validate_base_url(base_input or configured_url)
        except ValueError as exc:
            console.print(f"[red]Invalid endpoint: {exc}[/red]")
            return None

    parsed_endpoint = urlsplit(base_url)
    endpoint_port = parsed_endpoint.port or (
        443 if parsed_endpoint.scheme == "https" else 80
    )
    while not check_connection(
        parsed_endpoint.hostname or "", endpoint_port, connection_timeout
    ):
        choice = console.input(
            "[yellow]The selected API endpoint is not reachable. Restore access and "
            "retry (r), or return to local model selection (l)? [/yellow]"
        ).strip().lower()
        if choice != "r":
            return None

    host = parsed_endpoint.netloc
    if not use_configured and console.input(
        f"[yellow]Online mode will send requests to {host}. Continue? \\[y/N]: [/yellow]"
    ).strip().lower() != "y":
        return None

    api_key = configured_key if use_configured else get_api_key(
        "API key (input hidden; used for this session only): "
    ).strip()
    if not api_key:
        console.print("[red]No API key entered; returning to model selection.[/red]")
        return None

    model_choices: List[str] = []
    try:
        import httpx

        with public_sync_client(
            headers={"Authorization": "Bearer " + api_key},
            timeout=model_list_timeout,
            allow_loopback=True,
        ) as client:
            response = client.get(f"{base_url}/models")
            response.raise_for_status()
            body = response.json()
        model_choices = [
            item["id"] for item in body.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    except httpx.HTTPStatusError as exc:
        log_event(
            RUN_LOGGER,
            "provider.online_model_listing_failed",
            level=logging.WARNING,
            status_code=exc.response.status_code,
            error_type=type(exc).__name__,
        )
        if exc.response.status_code in {401, 403}:
            console.print(
                "[red]Provider authentication failed while listing models. "
                "The API key was not logged; verify the endpoint and key.[/red]"
            )
            return None
        console.print(
            f"[yellow]Could not list provider models (HTTP "
            f"{exc.response.status_code}); you can enter a model identifier manually.[/yellow]"
        )
    except Exception as exc:
        log_event(
            RUN_LOGGER,
            "provider.online_model_listing_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(
            f"[yellow]Could not list provider models ({type(exc).__name__}); "
            "you can enter a model identifier manually.[/yellow]"
        )

    configured_model_value = model_config.get("online_model")
    if configured_model_value is None:
        configured_model_value = app_config.get("online_model")
    configured_model = str(configured_model_value or "")
    if model_choices:
        console.print("[bold]Available API models:[/bold]")
        for index, candidate in enumerate(model_choices[:model_list_limit], 1):
            console.print(f"  [cyan]{index}.[/cyan] {candidate}")
        default_model = (
            configured_model if configured_model in model_choices else model_choices[0]
        )
        model_input = console.input(
            f"[cyan]Select model number or enter ID [{default_model}]: [/cyan]"
        ).strip()
        if model_input.isdigit() and 1 <= int(model_input) <= min(
            model_list_limit, len(model_choices)
        ):
            model_name = model_choices[int(model_input) - 1]
        else:
            model_name = model_input or default_model
    else:
        model_name = console.input(
            f"[cyan]Model identifier [{configured_model or 'enter model name'}]: [/cyan]"
        ).strip() or configured_model
    if not model_name:
        console.print("[red]A model identifier is required.[/red]")
        return None

    console.print(
        "[yellow]Every user message is sent to this online provider. Local chat "
        "history, summaries, skills, and RAG documents are excluded unless you "
        "opt in. Local tool calls can send tool arguments/results to this provider.[/yellow]"
    )
    share_context = console.input(
        "[yellow]Allow this online model to receive local history and retrieved "
        "RAG context for this session? \\[y/N]: [/yellow]"
    ).strip().lower() == "y"
    if permission_override:
        allow_tools = True
        console.print(
            "[yellow][Online permission override] All tools (including local-only "
            "location and media tools) are enabled for this online session; their "
            "arguments and results are sent to the provider. Workspace and "
            "permission-mode approvals still apply.[/yellow]"
        )
    else:
        allow_tools = console.input(
            "[yellow]Allow this online model to call local/MCP tools? Their arguments "
            "and results will be sent to the provider. \\[y/N]: [/yellow]"
        ).strip().lower() == "y"

    enforce_tool_call_limits = False
    if allow_tools:
        enforce_tool_call_limits = console.input(
            "[yellow]Apply configured limits to this online model "
            f"({max_tool_iterations} tool rounds and {max_tool_calls} tool calls "
            "per request)? Choose No to leave counts uncapped; request time and "
            "permission checks still apply. \\[y/N]: [/yellow]"
        ).strip().lower() == "y"

    try:
        model = model_factory(base_url, model_name, api_key)
    except Exception as exc:
        log_event(
            RUN_LOGGER,
            "provider.online_model_initialization_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(
            f"[red]Could not initialize online model ({type(exc).__name__}); "
            "check the endpoint, model, and key. Key details were not logged.[/red]"
        )
        return None
    return {
        "model": model,
        "model_name": model_name,
        "base_url": base_url,
        "share_context": share_context,
        "allow_tools": allow_tools,
        "enforce_tool_call_limits": enforce_tool_call_limits,
        "capabilities": online_capabilities(model, allow_tools),
    }


def online_capabilities(model, allow_tools: bool) -> dict:
    """Provider-neutral capabilities from LangChain's model profile.

    The profile is keyed by model id, not by endpoint, so it works for any
    OpenAI-compatible provider whose model id is recognised. Missing or
    unrecognised data stays None (unknown) rather than being guessed.
    """
    try:
        profile = getattr(model, "profile", None)
    except Exception as exc:
        log_event(
            RUN_LOGGER,
            "provider.capability_profile_unavailable",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        profile = None
    profile = profile if isinstance(profile, dict) else {}

    def declared(key):
        value = profile.get(key)
        return value if isinstance(value, bool) else None

    tool_calling = declared("tool_calling")
    return {
        "tools": False if not allow_tools else tool_calling,
        "function_calls": False if not allow_tools else tool_calling,
        "structured_output": declared("structured_output"),
        "thinking": declared("reasoning_output"),
        "vision": declared("image_inputs"),
        "audio": declared("audio_inputs"),
    }
