"""Providers command for extty-infra CLI."""

import typer
from rich.console import Console
from rich.table import Table

from extty_infra.config import get_provider_api_key, load_config
from extty_infra.providers.models import ProviderName

app = typer.Typer()
console = Console()

SUPPORTED_PROVIDERS: list[tuple[ProviderName, str, str]] = [
    ("lambda", "Lambda Cloud", "https://cloud.lambdalabs.com"),
    ("vast", "Vast.ai", "https://vast.ai"),
    ("prime", "Prime Intellect", "https://primeintellect.ai"),
]


@app.callback(invoke_without_command=True)
def list_providers() -> None:
    """List supported cloud providers and their configuration status."""
    config = load_config()

    table = Table(title="Cloud Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Default")

    for name, display_name, url in SUPPORTED_PROVIDERS:
        api_key = get_provider_api_key(name)

        if api_key:
            status = "[green]configured[/green]"
        else:
            status = "[dim]not configured[/dim]"

        is_default = "[yellow]✓[/yellow]" if name == config.default_provider else ""

        table.add_row(name, display_name, status, is_default)

    console.print(table)
    console.print()
    console.print("[dim]To configure a provider:[/dim]")
    console.print("  extty-infra config set <provider>.api_key <key>")
    console.print()
    console.print("[dim]To set default provider:[/dim]")
    console.print("  extty-infra config set default_provider <provider>")
