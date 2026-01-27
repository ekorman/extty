"""Config commands for extty-infra CLI."""

import typer
from rich.console import Console
from rich.table import Table

from extty_infra.config import load_config, set_config_value

app = typer.Typer(help="Manage configuration")
console = Console()


@app.command("set")
def config_set(
    key: str = typer.Argument(
        ..., help="Configuration key to set (e.g., api_key or vast.api_key)"
    ),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print(f"[green]Set {key} = {value}[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("show")
def config_show() -> None:
    """Display current configuration."""
    config = load_config()

    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("default_provider", config.default_provider)

    for key in ["api_key", "default_region", "default_instance_type", "ssh_key_name"]:
        value = getattr(config, key)
        display_value = value if value else "[dim]not set[/dim]"
        if key == "api_key" and value:
            display_value = value[:8] + "..." + value[-4:]
        table.add_row(key, display_value)

    for provider in ["lambda", "vast", "prime"]:
        if provider == "lambda":
            provider_config = config.lambda_config
        else:
            provider_config = getattr(config, provider)

        has_config = any(
            getattr(provider_config, k) is not None
            for k in [
                "api_key",
                "default_region",
                "default_instance_type",
                "ssh_key_name",
            ]
        )

        if has_config:
            table.add_row("", "")
            table.add_row(f"[bold]{provider}[/bold]", "")

            for key in [
                "api_key",
                "default_region",
                "default_instance_type",
                "ssh_key_name",
            ]:
                value = getattr(provider_config, key)
                if value is not None:
                    display_value = value
                    if key == "api_key":
                        display_value = (
                            value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                        )
                    table.add_row(f"  {key}", display_value)

    console.print(table)
