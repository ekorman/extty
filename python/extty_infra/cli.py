"""Main CLI entrypoint for extty-infra."""

import typer

from extty_infra.commands import config, instance, providers, run, types

app = typer.Typer(
    name="extty-infra",
    help="Cloud VM Management CLI (Lambda, Vast.ai, Prime Intellect)",
    no_args_is_help=True,
)

app.add_typer(config.app, name="config")
app.add_typer(types.app, name="types")
app.add_typer(providers.app, name="providers")
app.add_typer(instance.app, name="instance", hidden=True)

app.command("launch")(instance.launch)
app.command("list")(instance.list_instances)
app.command("terminate")(instance.terminate)
app.command("ssh")(instance.ssh)
app.command("run")(run.run)


if __name__ == "__main__":
    app()
