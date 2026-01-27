"""Run command for extty-infra CLI."""

import typer
from rich.console import Console

from extty_infra.config import get_provider_api_key, load_config
from extty_infra.providers import ProviderAPIError, get_provider
from extty_infra.ssh import rsync_to_remote, run_remote, wait_for_ssh
from extty_infra.state import load_state

app = typer.Typer()
console = Console()

REMOTE_CODE_DIR = "~/code"
UV_INSTALL_CMD = "command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)"


@app.callback(invoke_without_command=True)
def run(
    instance_id: str = typer.Argument(..., help="Instance ID to run on"),
    path: str = typer.Option(..., "--path", "-p", help="Local path to sync"),
    script: str = typer.Option(..., "--script", "-s", help="Script to run"),
    args: str | None = typer.Option(
        None, "--args", "-a", help="Arguments to pass to script"
    ),
) -> None:
    """Sync code and run a script on a remote instance."""
    state = load_state()
    instance = next((i for i in state.instances if i.instance_id == instance_id), None)

    if instance:
        if not instance.ip:
            console.print("[red]Error: Instance has no IP address.[/red]")
            raise typer.Exit(1)
        ip = instance.ip
        ssh_user = instance.ssh_user
    else:
        config = load_config()
        prov = config.default_provider

        api_key = get_provider_api_key(prov)
        if not api_key:
            console.print(f"[red]Error: API key not set for {prov}.[/red]")
            raise typer.Exit(1)

        try:
            with get_provider(prov, api_key) as client:
                api_instance = client.get_instance(instance_id)
                if not api_instance.ip:
                    console.print("[red]Error: Instance has no IP address.[/red]")
                    raise typer.Exit(1)
                ip = api_instance.ip
                ssh_user = api_instance.ssh_user
        except ProviderAPIError as e:
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1)

    console.print(
        f"[yellow]Checking SSH connectivity to {ip} as {ssh_user}...[/yellow]"
    )
    if not wait_for_ssh(ip, user=ssh_user, timeout=60, interval=2):
        console.print("[red]Error: Could not connect via SSH.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Syncing {path} to {ip}:{REMOTE_CODE_DIR}...[/yellow]")
    result = rsync_to_remote(path, ip, REMOTE_CODE_DIR, user=ssh_user)
    if result.returncode != 0:
        console.print("[red]Error: rsync failed.[/red]")
        raise typer.Exit(1)

    console.print("[yellow]Ensuring uv is installed...[/yellow]")
    result = run_remote(ip, UV_INSTALL_CMD, user=ssh_user, stream=False)
    if result.returncode != 0:
        console.print("[red]Error: Failed to install uv.[/red]")
        raise typer.Exit(1)

    script_args = args if args else ""
    run_cmd = (
        f"cd {REMOTE_CODE_DIR} && ~/.local/bin/uv run python {script} {script_args}"
    )

    console.print(f"[green]Running: {script} {script_args}[/green]")
    console.print("[dim]" + "-" * 60 + "[/dim]")

    result = run_remote(ip, run_cmd, user=ssh_user, stream=True)

    console.print("[dim]" + "-" * 60 + "[/dim]")
    if result.returncode == 0:
        console.print("[green]Script completed successfully.[/green]")
    else:
        console.print(f"[red]Script exited with code {result.returncode}.[/red]")
        raise typer.Exit(result.returncode)
