"""Types command for extty-infra CLI."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from extty_infra.config import get_provider_api_key, load_config
from extty_infra.providers import ProviderAPIError, ProviderName, get_provider

app = typer.Typer()
console = Console()

ProviderOption = Annotated[
    str | None,
    typer.Option(
        "--provider",
        "-p",
        help="Cloud provider (lambda, vast, prime). Defaults to config default_provider.",
    ),
]


def resolve_provider(provider_opt: str | None) -> ProviderName:
    """Resolve provider option to a valid ProviderName."""
    if provider_opt is None:
        config = load_config()
        return config.default_provider
    if provider_opt not in ("lambda", "vast", "prime"):
        console.print(
            f"[red]Error: Invalid provider '{provider_opt}'. Must be lambda, vast, or prime.[/red]"
        )
        raise typer.Exit(1)
    return provider_opt  # type: ignore[return-value]


@app.callback(invoke_without_command=True)
def list_types(
    provider: ProviderOption = None,
) -> None:
    """List available instance types."""
    prov = resolve_provider(provider)

    api_key = get_provider_api_key(prov)
    if not api_key:
        console.print(
            f"[red]Error: API key not set for {prov}. "
            f"Run 'extty-infra config set {prov}.api_key <key>'[/red]"
        )
        raise typer.Exit(1)

    try:
        with get_provider(prov, api_key) as client:
            instance_types = client.list_instance_types()
    except ProviderAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Available Instance Types ({prov})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("GPUs", justify="right")
    table.add_column("GPU Type")
    table.add_column("vCPUs", justify="right")
    table.add_column("Memory (GB)", justify="right")
    table.add_column("Storage (GB)", justify="right")
    table.add_column("$/hr", justify="right")
    table.add_column("Regions")

    for it in sorted(instance_types, key=lambda x: x.price_cents_per_hour):
        regions = ", ".join(it.regions[:3])
        if len(it.regions) > 3:
            regions += f" (+{len(it.regions) - 3})"

        table.add_row(
            it.name,
            it.description or "",
            str(it.gpu_count),
            it.gpu_name or "",
            str(it.vcpus),
            str(it.memory_gib),
            str(it.storage_gib),
            f"${it.price_cents_per_hour / 100:.2f}",
            regions if regions else "[dim]none[/dim]",
        )

    console.print(table)
