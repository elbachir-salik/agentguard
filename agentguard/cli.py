from __future__ import annotations

import os
import sys

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentguard.export import session_to_json
from agentguard.models import SessionRecord
from agentguard.storage import Storage

console = Console(force_terminal=True)


def _status_color(status: str) -> str:
    return {"completed": "green", "tripped": "red", "error": "yellow"}.get(status, "white")


def _parse_meta_options(meta: tuple[str, ...]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in meta:
        if "=" not in item:
            raise click.BadParameter(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter(f"Metadata key cannot be empty: {item}")
        filters[key] = value.strip()
    return filters


def _format_metadata(metadata: dict) -> str:
    if not metadata:
        return ""
    return ", ".join(f"{k}={v}" for k, v in metadata.items())


def _load_session(storage: Storage, session_id: str) -> SessionRecord | None:
    matches = storage.find_sessions_by_prefix(session_id)

    if not matches:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return None
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous ID, {len(matches)} matches. Be more specific.[/yellow]")
        return None

    record = storage.get_session(matches[0]["session_id"])
    if not record:
        console.print("[red]Session not found.[/red]")
        return None
    return record


@click.group()
def main():
    """AgentGuard -- The black box + circuit breaker for AI agents."""
    pass


@main.command()
@click.option("--agent", default=None, help="Filter by agent name")
@click.option("--status", default=None, help="Filter by status (completed/tripped/error)")
@click.option("--parent", default=None, help="Filter by parent session ID (prefix match)")
@click.option("--meta", multiple=True, help="Filter by metadata key=value (repeatable)")
@click.option("--limit", default=20, help="Max sessions to show")
def sessions(agent, status, parent, meta, limit):
    """List recorded sessions."""
    storage = Storage()
    metadata = _parse_meta_options(meta) if meta else None
    rows = storage.list_sessions(
        agent_name=agent,
        status=status,
        metadata=metadata,
        parent_session_id_prefix=parent,
        limit=limit,
    )

    if not rows:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions", show_lines=False)
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Agent", style="bold")
    table.add_column("Parent", style="dim", no_wrap=True)
    table.add_column("Status")
    table.add_column("Metadata", style="dim")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Started", style="dim")

    for r in rows:
        color = _status_color(r["status"])
        table.add_row(
            r["session_id"],
            r["agent_name"],
            r.get("parent_session_id") or "",
            f"[{color}]{r['status']}[/{color}]",
            _format_metadata(r.get("metadata", {})),
            str(r.get("total_tokens", 0) or 0),
            f"${r.get('total_cost_usd', 0) or 0:.4f}",
            r["started_at"][:19],
        )

    console.print(table)


@main.command()
@click.argument("session_id")
def replay(session_id):
    """Replay a session turn by turn."""
    storage = Storage()
    record = _load_session(storage, session_id)
    if not record:
        return

    lines: list[str] = []
    for turn in record.turns:
        status_icon = "[green]OK[/green]" if turn.status == "success" else "[red]ERR[/red]"
        tok = turn.tokens_in + turn.tokens_out
        lines.append(
            f"  Turn {turn.turn_number}  {status_icon}  "
            f"{tok} tok  ${turn.cost_usd:.4f}  {turn.latency_ms:.0f}ms"
        )

        if turn.input_messages:
            for msg in turn.input_messages[:2]:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if content:
                    preview = content[:80] + ("..." if len(content) > 80 else "")
                    lines.append(f'    -> {role}: "{preview}"')

        for tc in turn.tool_calls:
            name = tc.get("name", "?")
            args = tc.get("arguments", "")
            if args:
                preview = str(args)[:60] + ("..." if len(str(args)) > 60 else "")
                lines.append(f"    <- tool_call: {name}({preview})")
            else:
                lines.append(f"    <- tool_call: {name}()")

        content = turn.output.get("content")
        if content and not turn.tool_calls:
            preview = content[:80] + ("..." if len(content) > 80 else "")
            lines.append(f'    <- "{preview}"')

        lines.append("")

    if record.breaker_event:
        be = record.breaker_event
        lines.append("  [red bold]== CIRCUIT BREAKER =========================[/red bold]")
        lines.append(f"  [red]  Rule:    {be.rule}[/red]")
        lines.append(f"  [red]  Cause:   {be.trigger}[/red]")
        lines.append("  [red bold]============================================[/red bold]")
        lines.append("")

    if record.metadata:
        lines.append(f"  Metadata: {_format_metadata(record.metadata)}")
        lines.append("")

    if record.parent_session_id:
        lines.append(f"  Parent session: {record.parent_session_id}")
        lines.append("")

    color = _status_color(record.status)
    status_label = record.status.title()
    lines.append(
        f"  [{color}]{status_label}[/{color}] -- "
        f"{len(record.turns)} turns -- "
        f"{record.total_tokens} tok -- "
        f"${record.total_cost_usd:.4f}"
    )

    title = f"agentguard -- {record.agent_name} -- {record.session_id}"
    panel = Panel("\n".join(lines), title=title, border_style="bold")
    console.print(panel)


@main.command()
@click.argument("session_id")
@click.option("--format", "fmt", type=click.Choice(["json"]), default="json", show_default=True)
@click.option("-o", "--output", default=None, help="Write to file instead of stdout")
def export(session_id, fmt, output):
    """Export a session to JSON for sharing or debugging."""
    storage = Storage()
    record = _load_session(storage, session_id)
    if not record:
        return

    if fmt != "json":
        console.print(f"[red]Unsupported format: {fmt}[/red]")
        raise SystemExit(1)

    payload = session_to_json(record)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
        console.print(f"[green]Exported session {record.session_id} to {output}[/green]")
    else:
        click.echo(payload)


@main.command()
@click.option("--agent", default=None, help="Filter by agent name")
def stats(agent):
    """Show cost and usage stats."""
    storage = Storage()
    s = storage.get_stats(agent_name=agent)

    table = Table(title="AgentGuard Stats", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Sessions", str(s["total_sessions"]))
    table.add_row("Completed", f"[green]{s['completed']}[/green]")
    table.add_row("Tripped", f"[red]{s['trips']}[/red]")
    table.add_row("Errors", f"[yellow]{s['errors']}[/yellow]")
    table.add_row("Total Tokens", f"{s['total_tokens']:,}")
    table.add_row("Total Cost", f"${s['total_cost_usd']:.4f}")

    console.print(table)


@main.command()
@click.option("--host", default="127.0.0.1", help="Dashboard host")
@click.option("--port", default=8585, help="Dashboard port")
def dashboard(host, port):
    """Launch the local web dashboard."""
    try:
        import uvicorn

        from agentguard.dashboard.app import app  # noqa: F811

        console.print(f"[bold green]Starting dashboard at http://{host}:{port}[/bold green]")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except ImportError:
        console.print(
            "[red]Dashboard dependencies missing. "
            "Install with: pip install agentguard[dashboard][/red]"
        )


if __name__ == "__main__":
    main()
