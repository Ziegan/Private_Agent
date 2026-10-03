import logging
import os
import pathlib
import time
from typing import Any, Optional

from rich.console import Console

from .config import DEBUG_LOG_ENABLED


RUN_LOGGER = logging.getLogger("private_agent.run")
RUN_LOGGER.setLevel(logging.INFO)
RUN_LOGGER.propagate = False


def start_debug_logging() -> Optional[pathlib.Path]:
    """Start a private per-run trace file when enabled by configuration."""
    if not DEBUG_LOG_ENABLED:
        return None
    app_dir = pathlib.Path.home() / ".private_agent"
    log_dir = app_dir / "logs"
    try:
        if app_dir.is_symlink() or log_dir.is_symlink():
            raise OSError("Refusing to write debug logs through a symbolic link.")
        app_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(log_dir, 0o700)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"private-agent-{timestamp}-{os.getpid()}.log"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        ))
        RUN_LOGGER.addHandler(handler)
        RUN_LOGGER.info("Run trace started")
        return path
    except OSError as exc:
        raise RuntimeError(f"Could not create debug log in {log_dir}: {exc}") from exc


def stop_debug_logging() -> None:
    for handler in RUN_LOGGER.handlers[:]:
        RUN_LOGGER.removeHandler(handler)
        handler.flush()
        handler.close()


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
    """Preserve Rich CLI rendering while mirroring interactions to the trace."""

    def __init__(self, base_console: Optional[Console] = None):
        self._base = base_console or Console()

    def print(self, *values: Any, **kwargs: Any) -> None:
        RUN_LOGGER.info("OUTPUT\n%s", render_console_values(*values, **kwargs))
        self._base.print(*values, **kwargs)

    def input(self, prompt: Any = "", **kwargs: Any) -> str:
        RUN_LOGGER.info(
            "INPUT REQUEST\n%s",
            render_console_values(prompt, end="", **kwargs),
        )
        value = self._base.input(prompt, **kwargs)
        RUN_LOGGER.info("INPUT RESPONSE\n%s", value)
        return value

    def status(self, status: Any, *args: Any, **kwargs: Any):
        RUN_LOGGER.info("STATUS %s", render_console_values(status, end=""))
        return self._base.status(status, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def install_logging_console_adapter(adapter: LoggingConsoleAdapter):
    """Install log-mirroring consoles and return their original values."""
    from . import agent, code_tasks, rag, skills, tools
    from .tools import media_tools

    modules = (agent, code_tasks, rag, skills, tools, media_tools)
    originals = [(module, module.console) for module in modules]
    for module in modules:
        module.console = adapter
    return originals


def restore_logging_consoles(originals) -> None:
    for module, console in originals:
        module.console = console
