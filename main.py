import pathlib
from src.agent import run_agent_cli
from rich.console import Console

console = Console()

def main():
    try:
        run_agent_cli()
    except KeyboardInterrupt:
        console.print("\n[yellow][Info] Interrupted by user.[/yellow]")
    finally:
        console.print("[cyan][Info] Cleanup completed cleanly. Goodbye![/cyan]")

if __name__ == "__main__":
    main()
