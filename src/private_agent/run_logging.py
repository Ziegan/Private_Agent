import contextvars
import json
import logging
import os
import pathlib
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from rich.console import Console

from .config import DEBUG_LOG_ENABLED, DEBUG_LOG_LEVEL


RUN_LOGGER = logging.getLogger("private_agent.run")
RUN_LOGGER.propagate = False
PACKAGE_LOGGER = logging.getLogger("private_agent")
PACKAGE_LOGGER.propagate = False

_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "private_agent_run_id",
    default=None,
)
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "private_agent_log_context",
    default={},
)
_ACTIVE_HANDLERS: list[logging.Handler] = []
_RUN_ID_TOKEN: Optional[contextvars.Token] = None

_SENSITIVE_FIELD = re.compile(
    r"(?:api[_-]?(?:key|token)|authorization|credential|password|secret|"
    r"(?:^|[_-])prompt(?:$|[_-])|(?:^|[_-])arguments?(?:$|[_-])|"
    r"(?:^|[_-])raw(?:$|[_-])|(?:^|[_-])content(?:$|[_-])|"
    r"(?:access|refresh|auth|bearer|id)[_-]?token|"
    r"^token$|token[_-]?(?:value|text|secret))",
    re.IGNORECASE,
)
_LABELLED_SECRET = re.compile(
    r"(?i)\b(api[_-]?(?:key|token)|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password|passwd|secret|authorization)\b"
    r"(\s*[:=]\s*|\s+)"
    r"([^\s,;]+)"
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat|xox[baprs])-[A-Za-z0-9_-]{12,}\b"),
)


def _redact_text(value: str, values: tuple[str, ...] = ()) -> str:
    redacted = _LABELLED_SECRET.sub(r"\1\2[REDACTED]", value)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    for secret in sorted((item for item in values if item), key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _safe_field_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value[:1000])
    if isinstance(value, (list, tuple)):
        return [_safe_field_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key): _safe_field_value(item)
            for key, item in list(value.items())[:20]
            if not _SENSITIVE_FIELD.search(str(key))
        }
    return type(value).__name__


class JsonLineFormatter(logging.Formatter):
    """Serialize diagnostic records as one private, redacted JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        current_context = _LOG_CONTEXT.get()
        redaction_values = getattr(record, "private_redactions", ())
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "message": _redact_text(record.getMessage(), redaction_values),
            "source": {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
                "module": record.module,
            },
            "process": record.process,
            "thread": record.thread,
            "run_id": _RUN_ID.get(),
        }
        if current_context:
            payload["context"] = _safe_field_value(current_context)
        event_fields = getattr(record, "event_fields", None)
        if event_fields:
            payload["fields"] = _safe_field_value(event_fields)
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": _redact_text(str(record.exc_info[1]), redaction_values),
                "traceback": _redact_text(
                    "".join(traceback.format_exception(*record.exc_info)),
                    redaction_values,
                ),
            }
        if record.stack_info:
            payload["stack"] = _redact_text(
                self.formatStack(record.stack_info),
                redaction_values,
            )
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    stacklevel: int = 1,
    exc_info: Any = None,
    redactions: tuple[str, ...] = (),
    **fields: Any,
) -> None:
    """Emit a named event with bounded fields, excluding sensitive field names."""
    safe_fields = {
        key: _safe_field_value(value)
        for key, value in fields.items()
        if not _SENSITIVE_FIELD.search(key)
    }
    logger.log(
        level,
        event.replace("_", " "),
        extra={
            "event": event,
            "event_fields": safe_fields,
            "private_redactions": redactions,
        },
        exc_info=exc_info,
        stacklevel=stacklevel + 1,
    )


@contextmanager
def logging_context(**values: Any) -> Iterator[None]:
    """Temporarily attach non-sensitive correlation metadata to emitted events."""
    current = _LOG_CONTEXT.get()
    updated = {**current, **values}
    token = _LOG_CONTEXT.set(updated)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def start_debug_logging() -> Optional[pathlib.Path]:
    """Start a private per-run JSONL trace file when enabled by configuration."""
    global _RUN_ID_TOKEN
    if not DEBUG_LOG_ENABLED:
        return None
    if _ACTIVE_HANDLERS:
        stop_debug_logging()
    app_dir = pathlib.Path.home() / ".private_agent"
    log_dir = app_dir / "logs"
    try:
        if app_dir.is_symlink() or log_dir.is_symlink():
            raise OSError("Refusing to write debug logs through a symbolic link.")
        app_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(log_dir, 0o700)
        run_id = uuid.uuid4().hex
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"private-agent-{timestamp}-{os.getpid()}-{run_id[:8]}.log"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLineFormatter())
        level = getattr(logging, DEBUG_LOG_LEVEL, logging.INFO)
        handler.setLevel(level)
        for logger in (PACKAGE_LOGGER, RUN_LOGGER):
            logger.setLevel(level)
            logger.addHandler(handler)
        _ACTIVE_HANDLERS.append(handler)
        _RUN_ID_TOKEN = _RUN_ID.set(run_id)
        log_event(
            RUN_LOGGER,
            "run.started",
            run_id=run_id,
            log_level=DEBUG_LOG_LEVEL,
        )
        return path
    except OSError as exc:
        raise RuntimeError(f"Could not create debug log in {log_dir}: {exc}") from exc


def stop_debug_logging() -> None:
    global _RUN_ID_TOKEN
    for handler in _ACTIVE_HANDLERS:
        for logger in (RUN_LOGGER, PACKAGE_LOGGER):
            if handler in logger.handlers:
                logger.removeHandler(handler)
        handler.flush()
        handler.close()
    _ACTIVE_HANDLERS.clear()
    if _RUN_ID_TOKEN is not None:
        _RUN_ID.reset(_RUN_ID_TOKEN)
        _RUN_ID_TOKEN = None


def render_console_values(*values: Any, **kwargs: Any) -> str:
    import io

    output = io.StringIO()
    kwargs.pop("file", None)
    kwargs.pop("soft_wrap", None)
    render_console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=120,
        highlight=False,
    )
    render_console.print(*values, **kwargs)
    return output.getvalue().rstrip()


class LoggingConsoleAdapter:
    """Preserve Rich CLI rendering without recording conversation content."""

    def __init__(self, base_console: Optional[Console] = None):
        self._base = base_console or Console()

    def print(self, *values: Any, **kwargs: Any) -> None:
        rendered = render_console_values(*values, **kwargs)
        log_event(
            RUN_LOGGER,
            "console.output",
            stacklevel=2,
            character_count=len(rendered),
        )
        self._base.print(*values, **kwargs)

    def input(self, prompt: Any = "", **kwargs: Any) -> str:
        rendered_prompt = render_console_values(prompt, end="", **kwargs)
        log_event(
            RUN_LOGGER,
            "console.input_requested",
            stacklevel=2,
            request_character_count=len(rendered_prompt),
        )
        value = self._base.input(prompt, **kwargs)
        log_event(
            RUN_LOGGER,
            "console.input_received",
            stacklevel=2,
            response_character_count=len(value),
        )
        return value

    def status(self, status: Any, *args: Any, **kwargs: Any):
        rendered_status = render_console_values(status, end="")
        log_event(
            RUN_LOGGER,
            "console.status",
            stacklevel=2,
            status_character_count=len(rendered_status),
        )
        return self._base.status(status, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def install_logging_console_adapter(adapter: LoggingConsoleAdapter):
    """Install log-mirroring consoles and return their original values."""
    from . import code_tasks, tools
    from .agent import runtime
    from .rag import indexing, retrieval
    from .skills import loader
    from .tools import media, mcp, network, shell

    modules = (
        runtime,
        code_tasks,
        indexing,
        retrieval,
        loader,
        tools,
        media,
        mcp,
        network,
        shell,
    )
    originals = [(module, module.console) for module in modules]
    for module in modules:
        module.console = adapter
    return originals


def restore_logging_consoles(originals) -> None:
    for module, console in originals:
        module.console = console
