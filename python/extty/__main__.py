"""CLI entry point for extty."""

import argparse
import sys

from extty.sync import push, list_local_runs


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="extty",
        description="extty ML experiment tracker CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    push_parser = subparsers.add_parser("push", help="Push a run to remote server")
    push_parser.add_argument("run_name", nargs="?", help="Name of run to push")
    push_parser.add_argument("--server", "-s", help="Server URL")
    push_parser.add_argument("--all", action="store_true", help="Push all runs")

    list_parser = subparsers.add_parser("list", help="List local runs")

    args = parser.parse_args()

    if args.command == "push":
        if args.all:
            runs = list_local_runs()
            if not runs:
                print("No local runs found.")
                return 0
            for run_name in runs:
                try:
                    push(run_name, server_url=args.server)
                except Exception as e:
                    print(f"Failed to push '{run_name}': {e}")
        elif args.run_name:
            try:
                push(args.run_name, server_url=args.server)
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            print("Error: run_name required (or use --all)")
            return 1

    elif args.command == "list":
        runs = list_local_runs()
        if not runs:
            print("No local runs found.")
        else:
            print("Local runs:")
            for name in sorted(runs):
                print(f"  {name}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
