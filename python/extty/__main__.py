"""CLI entry point for extty."""

import argparse
import sys

from extty.sync import push, list_local_runs


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="extty",
        description="extty ML experiment tracker CLI",
    )

    # Add top-level --server argument for SSH access
    parser.add_argument(
        "--server",
        help="SSH server to read runs from (format: user@host or host)",
    )

    subparsers = parser.add_subparsers(dest="command")

    push_parser = subparsers.add_parser("push", help="Push a run to remote server")
    push_parser.add_argument("run_name", nargs="?", help="Name of run to push")
    push_parser.add_argument("--push-server", "-s", help="Server URL to push to")
    push_parser.add_argument("--all", action="store_true", help="Push all runs")

    list_parser = subparsers.add_parser("list", help="List runs")

    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("run_name", help="Name of run to show")

    args = parser.parse_args()

    if args.command == "push":
        # Determine which runs to push from
        if args.server:
            # List runs from SSH server
            try:
                from extty.ssh_sync import list_remote_runs
                runs = list_remote_runs(args.server)
            except Exception as e:
                print(f"Error listing remote runs: {e}")
                return 1
        else:
            # List local runs
            runs = list_local_runs()

        if args.all:
            if not runs:
                location = f"on {args.server}" if args.server else "locally"
                print(f"No runs found {location}.")
                return 0
            for run_name in runs:
                try:
                    push(run_name, server_url=args.push_server)
                except Exception as e:
                    print(f"Failed to push '{run_name}': {e}")
        elif args.run_name:
            try:
                push(args.run_name, server_url=args.push_server)
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            print("Error: run_name required (or use --all)")
            return 1

    elif args.command == "list":
        if args.server:
            # List runs from SSH server
            try:
                from extty.ssh_sync import list_remote_runs
                runs = list_remote_runs(args.server)
                if not runs:
                    print(f"No runs found on {args.server}.")
                else:
                    print(f"Runs on {args.server}:")
                    for name in sorted(runs):
                        print(f"  {name}")
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            # List local runs
            runs = list_local_runs()
            if not runs:
                print("No local runs found.")
            else:
                print("Local runs:")
                for name in sorted(runs):
                    print(f"  {name}")

    elif args.command == "show":
        if args.server:
            # Show run from SSH server
            try:
                from extty.ssh_sync import stream_remote_run_data
                import json
                data = stream_remote_run_data(args.server, args.run_name)
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            # Show local run
            try:
                from extty.sync import _build_payload
                from extty.storage import get_runs_dir
                import json
                run_dir = get_runs_dir() / args.run_name
                if not run_dir.exists():
                    print(f"Run '{args.run_name}' not found.")
                    return 1
                data = _build_payload(run_dir)
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error: {e}")
                return 1

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
