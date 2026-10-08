import argparse
import logging
import time

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        prog="private-agent",
        description="Run the local-first Private Agent CLI.",
    )
    parser.add_argument(
        "--verbose-startup",
        action="store_true",
        help="Show detailed startup diagnostics and configured paths.",
    )
    args = parser.parse_args()
    originals = []
    exit_code = 0
    from .config import DEBUG_LOG_ENABLED
    from .run_logging import (
        LoggingConsoleAdapter,
        RUN_LOGGER,
        install_logging_console_adapter,
        log_event,
        start_debug_logging,
        stop_debug_logging,
    )

    try:
        startup_started = time.monotonic()
        log_path = start_debug_logging()
        if DEBUG_LOG_ENABLED and log_path is None:
            console.print("[yellow][Debug log] Logging is enabled but no file was created.[/yellow]")
        elif log_path is not None:
            console.print(f"[cyan][Debug log][/cyan] Writing run trace to {log_path}")
        from .agent import run_agent_cli

        if log_path is not None:
            originals = install_logging_console_adapter(LoggingConsoleAdapter())
        log_event(RUN_LOGGER, "application.started")
        log_event(
            RUN_LOGGER,
            "application.startup_completed",
            elapsed_seconds=round(time.monotonic() - startup_started, 3),
        )
        run_agent_cli(verbose_startup=args.verbose_startup)
    except KeyboardInterrupt:
        console.print("\n[yellow][Info] Interrupted by user.[/yellow]")
    except RuntimeError as exc:
        log_event(
            RUN_LOGGER,
            "application.startup_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(f"[red][Startup error] {exc}[/red]")
        exit_code = 1
    except Exception:
        log_event(
            RUN_LOGGER,
            "application.terminated_unexpectedly",
            level=logging.ERROR,
            exc_info=True,
        )
        raise
    finally:
        log_event(RUN_LOGGER, "application.exiting")
        if originals:
            from .run_logging import restore_logging_consoles

            restore_logging_consoles(originals)
        stop_debug_logging()
        console.print("[cyan][Info] Goodbye.[/cyan]")
    if exit_code:
        raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
