import logging
import os
import pathlib
import time
from typing import Any, Optional

from rich.console import Console

from .config import DEBUG_LOG_ENABLED, DEBUG_LOG_LEVEL


RUN_LOGGER = logging.getLogger("private_agent.run")
RUN_LOGGER.propagate = False
# Package logger: module loggers (getLogger(__name__)) reach the trace file
# through it, but never the terminal, because it only has the file handler.
PACKAGE_LOGGER = logging.getLogger("private_agent")
PACKAGE_LOGGER.propagate = False
LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        level = getattr(logging, DEBUG_LOG_LEVEL, logging.INFO)
        handler.setLevel(level)
        for logger in (PACKAGE_LOGGER, RUN_LOGGER):
            logger.setLevel(level)
            logger.addHandler(handler)
        RUN_LOGGER.info("Run trace started (level %s)", DEBUG_LOG_LEVEL)
        return path
    except OSError as exc:
        raise RuntimeError(f"Could not create debug log in {log_dir}: {exc}") from exc


def stop_debug_logging() -> None:
    for logger in (RUN_LOGGER, PACKAGE_LOGGER):
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
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
        RUN_LOGGER.info(
            "OUTPUT\n%s", render_console_values(*values, **kwargs), stacklevel=2
        )
        self._base.print(*values, **kwargs)

    def input(self, prompt: Any = "", **kwargs: Any) -> str:
        RUN_LOGGER.info(
            "INPUT REQUEST\n%s",
            render_console_values(prompt, end="", **kwargs),
            stacklevel=2,
        )
        value = self._base.input(prompt, **kwargs)
        RUN_LOGGER.info("INPUT RESPONSE\n%s", value, stacklevel=2)
        return value

    def status(self, status: Any, *args: Any, **kwargs: Any):
        RUN_LOGGER.info(
            "STATUS %s", render_console_values(status, end=""), stacklevel=2
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
