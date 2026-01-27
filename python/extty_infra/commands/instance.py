"""Instance commands for extty-infra CLI."""

from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from extty_infra.config import get_provider_api_key, load_config
from extty_infra.models import InstanceState
from extty_infra.providers import (
    CloudProvider,
    ProviderAPIError,
    ProviderLaunchOptions,
    ProviderName,
    get_provider,
)
from extty_infra.ssh import interactive_ssh, wait_for_ssh
from extty_infra.state import (
    add_instance,
    get_instance,
    load_state,
    sync_with_provider,
)

app = typer.Typer(help="Manage instances")
console = Console()

ProviderOption = Annotated[
    str | None,
    typer.Option(
        "--provider",
        "-p",
        help="Cloud provider (lambda, vast, prime). Defaults to config default_provider.",
    ),
]


def get_provider_client(provider_name: ProviderName | None = None) -> CloudProvider:
    """Get an authenticated provider client."""
    config = load_config()
    provider = provider_name or config.default_provider

    api_key = get_provider_api_key(provider)
    if not api_key:
        console.print(
            f"[red]Error: API key not set for {provider}. "
            f"Run 'extty-infra config set {provider}.api_key <key>'[/red]"
        )
        raise typer.Exit(1)

    return get_provider(provider, api_key)


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


@app.command("list")
def list_instances(
    provider: ProviderOption = None,
    all_providers: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="List instances from all configured providers"
        ),
    ] = False,
) -> None:
    """List running instances."""
    if all_providers:
        all_instances: list[InstanceState] = []
        for prov in ("lambda", "vast", "prime"):
            api_key = get_provider_api_key(prov)  # type: ignore[arg-type]
            if not api_key:
                continue
            try:
                with get_provider(prov, api_key) as client:  # type: ignore[arg-type]
                    api_instances = client.list_instances()
                    instances = sync_with_provider(prov, api_instances)  # type: ignore[arg-type]
                    all_instances.extend(instances)
            except ProviderAPIError as e:
                console.print(
                    f"[yellow]Warning: Could not fetch from {prov}: {e}[/yellow]"
                )
        instances = all_instances
    else:
        prov = resolve_provider(provider)
        try:
            with get_provider_client(prov) as client:
                api_instances = client.list_instances()
                instances = sync_with_provider(prov, api_instances)
        except ProviderAPIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1)

    if not instances:
        console.print("[dim]No instances found.[/dim]")
        return

    table = Table(title="Instances")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Region")
    table.add_column("IP")
    table.add_column("Status")

    for inst in instances:
        status_style = {
            "running": "green",
            "booting": "yellow",
            "pending": "yellow",
            "terminated": "red",
            "stopping": "red",
            "stopped": "dim",
            "error": "red",
        }.get(inst.status, "white")

        table.add_row(
            inst.instance_id,
            inst.provider,
            inst.name or "[dim]unnamed[/dim]",
            inst.instance_type,
            inst.region,
            inst.ip or "[dim]pending[/dim]",
            f"[{status_style}]{inst.status}[/{status_style}]",
        )

    console.print(table)


@app.command("launch")
def launch(
    provider: ProviderOption = None,
    instance_type: str | None = typer.Option(
        None, "--type", "-t", help="Instance type (e.g., gpu_1x_a10)"
    ),
    region: str | None = typer.Option(None, "--region", "-r", help="Region name"),
    name: str | None = typer.Option(None, "--name", "-n", help="Instance name"),
    ssh_key: str | None = typer.Option(None, "--ssh-key", "-k", help="SSH key name"),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for SSH to be ready"
    ),
) -> None:
    """Launch a new instance."""
    prov = resolve_provider(provider)
    config = load_config()
    provider_config = config.get_provider_config(prov)

    instance_type = instance_type or provider_config.default_instance_type
    region = region or provider_config.default_region
    ssh_key = ssh_key or provider_config.ssh_key_name

    if not instance_type:
        console.print(
            f"[red]Error: Instance type required. Use --type or set {prov}.default_instance_type in config.[/red]"
        )
        raise typer.Exit(1)

    ssh_key_names = [ssh_key] if ssh_key else None
    if prov == "lambda" and not ssh_key_names:
        console.print(
            f"[red]Error: SSH key required for Lambda Cloud. Use --ssh-key or set {prov}.ssh_key_name in config.[/red]"
        )
        raise typer.Exit(1)

    if prov == "lambda" and not region:
        console.print(
            f"[red]Error: Region required for Lambda Cloud. Use --region or set {prov}.default_region in config.[/red]"
        )
        raise typer.Exit(1)

    options = ProviderLaunchOptions(
        instance_type=instance_type,
        region=region,
        ssh_key_names=ssh_key_names,
        name=name,
    )

    try:
        with get_provider_client(prov) as client:
            console.print(f"[yellow]Launching {instance_type} on {prov}...[/yellow]")
            response = client.launch(options)

            for instance_id in response.instance_ids:
                console.print(f"[green]Launched instance: {instance_id}[/green]")

                instance = client.get_instance(instance_id)
                instance_state = InstanceState(
                    instance_id=instance_id,
                    provider=prov,
                    name=name,
                    created_at=datetime.now(),
                    instance_type=instance_type,
                    region=instance.region,
                    ip=instance.ip,
                    status=instance.status,
                    ssh_user=client.ssh_user,
                )
                add_instance(instance_state)

                if wait and instance.ip:
                    console.print(
                        f"[yellow]Waiting for SSH on {instance.ip}...[/yellow]"
                    )
                    if wait_for_ssh(instance.ip, user=client.ssh_user):
                        console.print(
                            f"[green]SSH ready! Connect with: extty-infra ssh {instance_id}[/green]"
                        )
                    else:
                        console.print(
                            "[yellow]SSH not ready yet. Instance may still be booting.[/yellow]"
                        )

    except ProviderAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("terminate")
def terminate(
    instance_id: str = typer.Argument(..., help="Instance ID to terminate"),
    provider: ProviderOption = None,
) -> None:
    """Terminate an instance."""
    existing = get_instance(instance_id)
    if existing and provider is None:
        prov = existing.provider
    else:
        prov = resolve_provider(provider)

    try:
        with get_provider_client(prov) as client:
            console.print(
                f"[yellow]Terminating instance {instance_id} on {prov}...[/yellow]"
            )
            client.terminate([instance_id])
            console.print(f"[green]Instance {instance_id} terminated.[/green]")
    except ProviderAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("ssh")
def ssh(
    instance_id: str = typer.Argument(..., help="Instance ID to connect to"),
) -> None:
    """SSH into an instance."""
    state = load_state()
    instance = next((i for i in state.instances if i.instance_id == instance_id), None)

    if instance:
        if not instance.ip:
            console.print(
                "[red]Error: Instance has no IP address. It may still be booting.[/red]"
            )
            raise typer.Exit(1)
        ip = instance.ip
        ssh_user = instance.ssh_user
    else:
        config = load_config()
        prov = config.default_provider

        try:
            with get_provider_client(prov) as client:
                api_instance = client.get_instance(instance_id)
                if not api_instance.ip:
                    console.print("[red]Error: Instance has no IP address yet.[/red]")
                    raise typer.Exit(1)
                ip = api_instance.ip
                ssh_user = api_instance.ssh_user
        except ProviderAPIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Connecting to {ip} as {ssh_user}...[/green]")
    interactive_ssh(ip, user=ssh_user)
