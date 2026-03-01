#!/usr/bin/env python3
"""Sales Call AI Copilot — CLI entry point."""

import sys
import os
import io

# Ensure project root is on sys.path so `import config` / `import src.*` work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix SSL cert path for conda environments where SSL_CERT_FILE may be stale
if os.environ.get("SSL_CERT_FILE") and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        del os.environ["SSL_CERT_FILE"]

# Force UTF-8 on Windows so Unicode symbols (₹, —, etc.) render correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.chatbot.engine import ChatEngine

console = Console()

BANNER = """\
[bold cyan]Sales Call AI Copilot[/bold cyan]
[dim]RAG-powered assistant for sales call transcripts[/dim]

Commands you can try:
  [green]list my call ids[/green]          — show ingested calls
  [green]summarise call 1[/green]          — summarise a specific call
  [green]What pricing was discussed?[/green] — ask any question
  [green]ingest <path>[/green]             — add a new transcript
  [green]exit[/green]                      — quit
"""

EXIT_WORDS = {"exit", "quit", "bye", "q"}


def main():
    console.print(Panel(BANNER, title="Welcome", border_style="cyan"))

    engine = ChatEngine()

    # Auto-ingest transcripts if the store is empty
    if not engine.list_calls().startswith("Ingested"):
        console.print("[yellow]No calls found — auto-ingesting transcripts...[/yellow]")
        messages = engine.auto_ingest()
        for msg in messages:
            console.print(f"  {msg}")
        console.print()

    # REPL loop
    while True:
        try:
            user_input = console.input("[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            console.print("[dim]Goodbye![/dim]")
            break

        with console.status("[cyan]Thinking...[/cyan]"):
            response = engine.process_query(user_input)

        console.print()
        console.print(Panel(Markdown(response), title="Copilot", border_style="blue"))
        console.print()


if __name__ == "__main__":
    main()
