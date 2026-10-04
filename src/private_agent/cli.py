import argparse
import time

from rich.console import Console

console = Console()


def main():
    argparse.ArgumentParser(
        prog="private-agent",
        description="Run the local-first Private Agent CLI.",
    ).parse_args()
    originals = []
    exit_code = 0
    from .config import DEBUG_LOG_ENABLED
    from .run_logging import (
        LoggingConsoleAdapter,
        RUN_LOGGER,
        install_logging_console_adapter,
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
            RUN_LOGGER.info("CLI session started")
        RUN_LOGGER.info(
            "TIMING phase=process-startup elapsed=%.3fs",
            time.monotonic() - startup_started,
        )
        run_agent_cli()
    except KeyboardInterrupt:
        console.print("\n[yellow][Info] Interrupted by user.[/yellow]")
    except RuntimeError as exc:
        RUN_LOGGER.exception("Application startup failed")
        console.print(f"[red][Startup error] {exc}[/red]")
        exit_code = 1
    except Exception:
        RUN_LOGGER.exception("Application terminated unexpectedly")
        raise
    finally:
        RUN_LOGGER.info("Application exiting")
        if originals:
            from .run_logging import restore_logging_consoles

            restore_logging_consoles(originals)
        stop_debug_logging()
        console.print("[cyan][Info] Goodbye.[/cyan]")
    if exit_code:
        raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
