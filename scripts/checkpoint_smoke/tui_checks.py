"""
TUI side of the checkpoint smoke test, run on this machine after ``smoke.py``.

Reaches the test host's S3 server through an ssh tunnel and checks
``extty prune local``, ``extty pull``, the TUI's checkpoint download, and the
remote S3 config install against what ``smoke.py`` left behind. Needs only the
standard library. Usually driven by ``run.sh``.

Usage::

    python3 tui_checks.py --host HOST --remote-workdir DIR --extty BIN --tui-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TUNNEL_PORT = 19000
PASSED: list[str] = []
FAILED: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}", flush=True)
    (PASSED if ok else FAILED).append(what)


def open_tunnel(host: str) -> subprocess.Popen[bytes]:
    tunnel = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{TUNNEL_PORT}:127.0.0.1:9000",
            host,
        ]
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", TUNNEL_PORT), timeout=1).close()
            return tunnel
        except OSError:
            time.sleep(0.5)
    tunnel.terminate()
    raise RuntimeError("ssh tunnel to the S3 server did not come up")


def make_home(path: Path, summary: dict) -> Path:
    """An extty home whose S3 config points at the tunnelled server."""
    (path / "s3").mkdir(parents=True, exist_ok=True)
    (path / "s3" / "config.toml").write_text(
        f'bucket = "{summary["bucket"]}"\n'
        f'prefix = "{summary["prefix"]}"\n'
        'region = "us-east-1"\n'
        'access_key_id = "smoke"\n'
        'secret_access_key = "smoke-secret"\n'
        f'endpoint_url = "http://127.0.0.1:{TUNNEL_PORT}"\n'
    )
    return path


def extty(binary: str, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args],
        env={**os.environ, "EXTTY_HOME": str(home)},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )


def prune_marks(output: str, project: str) -> dict[tuple[str, int], str]:
    """Map (run, step) to the status text in ``extty prune local`` output."""
    marks: dict[tuple[str, int], str] = {}
    run = ""
    for line in output.splitlines():
        stripped = line.strip()
        if line.startswith("  ") and stripped.startswith(f"{project}/"):
            run = stripped.split("/", 1)[1]
        elif stripped.startswith("step ") and run:
            step = int(stripped.split()[1])
            marks[(run, step)] = stripped
    return marks


def check_prune(args: argparse.Namespace, summary: dict, tmp: Path) -> None:
    project = summary["project"]
    home = make_home(tmp / "prune-home", summary)
    subprocess.run(
        ["rsync", "-a", f"{args.host}:{summary['prune']['home']}/runs", str(home)],
        check=True,
    )

    def step_dir(run: str, step: int) -> Path:
        return home / "runs" / project / run / "checkpoints" / str(step)

    dry = extty(args.extty, home, "prune", "local", "--dry-run")
    marks = prune_marks(dry.stdout, project)
    expected = {
        ("round-trip", 1): "✓ in S3",
        ("old-run", 2): "✓ in S3 (old download)",
        ("no-bucket", 3): "not in S3 — skip",
        ("outage", 4): "S3 has a different save — skip",
    }
    for (run, step), mark in expected.items():
        check(
            mark in marks.get((run, step), ""),
            f"prune --dry-run: {run} step {step} shows '{mark}'",
        )
    if FAILED:
        print(dry.stdout, dry.stderr, sep="\n")

    pruned = extty(args.extty, home, "prune", "local", "--yes")
    check(pruned.returncode == 0, "prune: runs to completion")
    for run, step in summary["prune"]["deletable"]:
        check(not step_dir(run, step).exists(), f"prune: removed {run} step {step}")
    for run, step in summary["prune"]["kept"]:
        check(
            step_dir(run, step).exists(),
            f"prune: kept the only copy, {run} step {step}",
        )


def check_pull(args: argparse.Namespace, summary: dict, tmp: Path) -> None:
    project = summary["project"]
    home = make_home(tmp / "pull-home", summary)
    pulled = extty(args.extty, home, "pull", f"{project}/round-trip")
    run_dir = home / "runs" / project / "round-trip"
    check(pulled.returncode == 0, "pull: runs to completion")
    index = run_dir / "checkpoints.json"
    check(
        index.exists() and index.read_text() == summary["round_trip_index"],
        "pull: checkpoints.json is a verbatim copy of S3's index",
    )
    check(
        not (run_dir / "checkpoints").exists(),
        "pull: checkpoint files are not downloaded",
    )


def check_cargo_smoke_tests(args: argparse.Namespace, summary: dict) -> None:
    remote_dir = f"{args.remote_workdir}/ssh-config-check"
    env = {
        **os.environ,
        "EXTTY_SMOKE_ENDPOINT": f"http://127.0.0.1:{TUNNEL_PORT}",
        "EXTTY_SMOKE_BUCKET": summary["bucket"],
        "EXTTY_SMOKE_PREFIX": summary["prefix"],
        "EXTTY_SMOKE_SSH_HOST": args.host,
        "EXTTY_SMOKE_REMOTE_DIR": remote_dir,
    }
    for test, what in (
        (
            "s3::sync::tests::smoke_download_checkpoint",
            "TUI checkpoint download against live S3",
        ),
        (
            "run::tests::smoke_send_s3_config_over_ssh",
            "S3 config install over ssh via the remote login shell",
        ),
    ):
        result = subprocess.run(
            ["cargo", "test", "--quiet", "--", "--ignored", "--exact", test],
            cwd=args.tui_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        ok = result.returncode == 0 and "1 passed" in result.stdout
        check(ok, what)
        if not ok:
            print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")
    subprocess.run(["ssh", args.host, f"rm -rf {remote_dir}"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-workdir", required=True)
    parser.add_argument("--extty", required=True, help="path to the extty binary")
    parser.add_argument("--tui-dir", required=True)
    args = parser.parse_args()

    summary = json.loads(
        subprocess.run(
            ["ssh", args.host, f"cat {args.remote_workdir}/summary.json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    tunnel = open_tunnel(args.host)
    tmp = Path(tempfile.mkdtemp(prefix="extty-tui-smoke-"))
    try:
        print("--- prune", flush=True)
        check_prune(args, summary, tmp)
        print("--- pull", flush=True)
        check_pull(args, summary, tmp)
        print("--- cargo smoke tests", flush=True)
        check_cargo_smoke_tests(args, summary)
    finally:
        tunnel.terminate()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
