import argparse
from rich.console import Console

console = Console()

def main():
    parser = argparse.ArgumentParser(
        prog="private-agent",
        description="Run the local-first Private Agent CLI.",
    )
    parser.parse_args()
    try:
        from src.agent import run_agent_cli

        run_agent_cli()
    except KeyboardInterrupt:
        console.print("\n[yellow][Info] Interrupted by user.[/yellow]")
    finally:
        console.print("[cyan][Info] Goodbye.[/cyan]")

if __name__ == "__main__":
    main()
