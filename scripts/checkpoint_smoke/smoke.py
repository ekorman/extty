"""
End-to-end smoke test of extty checkpoints against a real S3 server.

Runs the SDK side of the checkpoint lifecycle with real torch, a real S3
endpoint, real processes, and checkpoints written by the ``main`` branch's SDK.
Usually driven by ``run.sh``, which also runs the TUI checks in
``tui_checks.py``.

Usage::

    PYTHONPATH=<new sdk> python smoke.py run \\
        --workdir DIR --endpoint URL --bucket NAME --container NAME \\
        --main-sdk <main sdk dir>

The S3 prefix is unique per run, so runs never see each other's data. Results
are printed as PASS / FAIL lines, and ``DIR/summary.json`` records what the
TUI checks need. Exits non-zero if any check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import torch

PROJECT = "smoke"
ACCESS_KEY = "smoke"
SECRET_KEY = "smoke-secret"
REGION = "us-east-1"


@dataclass
class Results:
    passed: int = 0
    failed: list[str] = field(default_factory=list)

    def check(self, ok: bool, what: str) -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {what}", flush=True)
        if ok:
            self.passed += 1
        else:
            self.failed.append(what)
        return ok


@dataclass
class Env:
    workdir: Path
    endpoint: str
    bucket: str
    prefix: str
    container: str
    main_sdk: Path
    results: Results

    def s3(self) -> Any:
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=REGION,
        )

    def step_prefix(self, run: str, step: int) -> str:
        return f"{self.prefix}/runs/{PROJECT}/{run}/checkpoints/{step}/"

    def step_keys(self, run: str, step: int) -> list[str]:
        prefix = self.step_prefix(run, step)
        pages = (
            self.s3()
            .get_paginator("list_objects_v2")
            .paginate(Bucket=self.bucket, Prefix=prefix)
        )
        return sorted(
            obj["Key"].removeprefix(prefix)
            for page in pages
            for obj in page.get("Contents", [])
        )

    def remote_meta(self, run: str, step: int) -> dict[str, Any]:
        body = self.s3().get_object(
            Bucket=self.bucket, Key=self.step_prefix(run, step) + "meta.json"
        )["Body"]
        return json.loads(body.read())

    def remote_index(self, run: str) -> bytes:
        key = f"{self.prefix}/runs/{PROJECT}/{run}/checkpoints.json"
        return self.s3().get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def home_path(self, name: str) -> Path:
        """
        The extty home ``name``, laid out as ``<name>/.extty``.

        ``main``'s SDK predates ``EXTTY_HOME`` and always uses ``~/.extty``, so
        its processes get ``HOME=<name>`` to land in the same place.
        """
        path = self.workdir / "homes" / name / ".extty"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def home(self, name: str) -> Path:
        """Switch this process to the extty home ``name``, returning it."""
        path = self.home_path(name)
        os.environ["EXTTY_HOME"] = str(path)
        return path

    def child(
        self, *args: str, home: Path, sdk: Path | None = None, **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Run a subcommand of this script in a fresh process using ``home``."""
        env = {
            **os.environ,
            "EXTTY_HOME": str(home),
            "HOME": str(home.parent),
            "PYTHONPATH": str(sdk) if sdk else os.environ["PYTHONPATH"],
        }
        return subprocess.run(
            [sys.executable, __file__, *args],
            env=env,
            capture_output=True,
            text=True,
            **kwargs,
        )


def configure_s3_env(endpoint: str, bucket: str, prefix: str) -> None:
    os.environ.update(
        {
            "EXTTY_S3_BUCKET": bucket,
            "EXTTY_S3_ENDPOINT_URL": endpoint,
            "EXTTY_S3_ACCESS_KEY_ID": ACCESS_KEY,
            "EXTTY_S3_SECRET_ACCESS_KEY": SECRET_KEY,
            "EXTTY_S3_REGION": REGION,
            "EXTTY_S3_PREFIX": prefix,
        }
    )


def model_state(seed: int, size: int = 64) -> tuple[dict[str, Any], dict[str, Any]]:
    """A trained-one-step Linear model's state dict and Adam state dict."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(size, size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.randn(8, size)).sum().backward()
    opt.step()
    return model.state_dict(), opt.state_dict()


def same(a: Any, b: Any) -> bool:
    """Structural equality for nested state dicts holding tensors."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(same(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def digest(state_dict: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].numpy().tobytes())
    return h.hexdigest()


def staging_entries(home: Path, run: str) -> list[Path]:
    staging = home / "runs" / PROJECT / run / "checkpoints" / ".staging"
    return sorted(staging.iterdir()) if staging.exists() else []


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def dead_endpoint_config(env: Env) -> Any:
    from extty.s3 import S3Config

    return S3Config(
        bucket=f"{env.bucket}-no-such-bucket",
        prefix=env.prefix,
        region=REGION,
        access_key_id=ACCESS_KEY,
        secret_access_key=SECRET_KEY,
        endpoint_url=env.endpoint,
    )


def wait_for_s3(env: Env, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            env.s3().head_bucket(Bucket=env.bucket)
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(1)


def scenario_round_trip(env: Env) -> None:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_meta
    from extty.s3 import S3Config
    from extty.storage import get_run_dir

    r = env.results
    env.home("trainer")
    run = extty.Run(PROJECT, name="round-trip", system_metrics=False)
    model, opt = model_state(seed=1)
    run.save_checkpoint(1, state_dict=model, optimizer_state_dict=opt)
    run.finish()

    meta = env.remote_meta("round-trip", 1)
    save_id = meta.get("save_id")
    r.check(bool(save_id), "round trip: S3 meta.json names a save_id")
    r.check(
        env.step_keys("round-trip", 1)
        == sorted(["meta.json", f"{save_id}/model.pt", f"{save_id}/optimizer.pt"]),
        "round trip: S3 files sit under the save's own prefix",
    )
    r.check(
        not local_checkpoint_dir(PROJECT, "round-trip", 1).exists(),
        "round trip: no local copy is kept after a successful upload",
    )
    mirror = get_run_dir(PROJECT, "round-trip") / "checkpoints.json"
    r.check(
        [e.get("save_id") for e in json.loads(mirror.read_text())] == [save_id],
        "round trip: local checkpoints.json mirrors the uploaded save",
    )

    env.home("reader")
    loaded = extty.load_checkpoint_from(PROJECT, "round-trip", 1)
    r.check(
        same(loaded["model_state_dict"], model)
        and same(loaded["optimizer_state_dict"], opt),
        "round trip: a fresh machine loads identical model and optimizer state",
    )
    local = read_meta(local_checkpoint_dir(PROJECT, "round-trip", 1))
    r.check(
        (local or {}).get("save_id") == save_id,
        "round trip: the download is committed with S3's save_id",
    )
    offline = S3Config(bucket=env.bucket, endpoint_url="http://127.0.0.1:1")
    again = extty.load_checkpoint_from(PROJECT, "round-trip", 1, s3_config=offline)
    r.check(
        same(again["model_state_dict"], model),
        "round trip: a second load works with S3 unreachable",
    )
    r.check(
        staging_entries(env.home_path("reader"), "round-trip") == [],
        "round trip: no staging leftovers after loading",
    )


def scenario_resave(env: Env) -> None:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_meta

    r = env.results
    env.home("trainer")
    run = extty.Run(PROJECT, name="resave", system_metrics=False)
    first, _ = model_state(seed=2)
    run.save_checkpoint(2, state_dict=first, keep_local=True)
    first_id = env.remote_meta("resave", 2)["save_id"]
    local = read_meta(local_checkpoint_dir(PROJECT, "resave", 2))
    r.check(
        (local or {}).get("save_id") == first_id,
        "re-save: keep_local keeps a committed copy of the uploaded save",
    )

    second, _ = model_state(seed=3)
    run.save_checkpoint(2, state_dict=second)
    second_id = env.remote_meta("resave", 2)["save_id"]
    r.check(second_id != first_id, "re-save: S3 now names the new save")
    r.check(
        env.step_keys("resave", 2) == sorted(["meta.json", f"{second_id}/model.pt"]),
        "re-save: the superseded save's files are gone from S3",
    )
    r.check(
        not local_checkpoint_dir(PROJECT, "resave", 2).exists(),
        "re-save: the older kept copy no longer shadows the new save",
    )
    r.check(
        same(run.load_checkpoint(2)["model_state_dict"], second),
        "re-save: loading returns the new weights",
    )
    run.finish()


def scenario_failed_upload(env: Env) -> None:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_local_checkpoint

    r = env.results
    env.home("trainer")
    capture = LogCapture()
    logging.getLogger("extty").addHandler(capture)
    try:
        run = extty.Run(
            PROJECT,
            name="no-bucket",
            system_metrics=False,
            s3_config=dead_endpoint_config(env),
        )
        model, _ = model_state(seed=4)
        run.save_checkpoint(3, state_dict=model)
        run.finish()
    finally:
        logging.getLogger("extty").removeHandler(capture)

    local = read_local_checkpoint(local_checkpoint_dir(PROJECT, "no-bucket", 3))
    r.check(
        local is not None and same(local["model_state_dict"], model),
        "failed upload (S3UploadFailedError): the save is kept locally and loads",
    )
    r.check(
        any("local copy kept at" in m for m in capture.messages),
        "failed upload: an error names the local copy",
    )
    try:
        extty.delete_local_checkpoint(PROJECT, "no-bucket", 3)
        refused = ""
    except RuntimeError as e:
        refused = str(e)
    r.check(
        "It is not in S3" in refused, "failed upload: deleting the only copy is refused"
    )
    listed = [
        c.step for c in extty.get_run(PROJECT, "no-bucket", local_only=True).checkpoints
    ]
    r.check(listed == [3], "failed upload: get_run still lists the local-only save")


def scenario_outage(env: Env) -> None:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_meta

    r = env.results
    env.home("trainer")
    run = extty.Run(PROJECT, name="outage", system_metrics=False)
    first, _ = model_state(seed=5)
    run.save_checkpoint(4, state_dict=first, keep_local=True)
    first_id = env.remote_meta("outage", 4)["save_id"]

    subprocess.run(
        ["docker", "stop", "-t", "1", env.container], check=True, capture_output=True
    )
    try:
        second, _ = model_state(seed=6)
        started = time.time()
        run.save_checkpoint(4, state_dict=second)
        print(
            f"      (save during outage took {time.time() - started:.0f}s)", flush=True
        )
    finally:
        subprocess.run(
            ["docker", "start", env.container], check=True, capture_output=True
        )
        wait_for_s3(env)
    run.finish()

    r.check(
        env.remote_meta("outage", 4)["save_id"] == first_id,
        "outage: S3 still holds the save from before the outage",
    )
    local = read_meta(local_checkpoint_dir(PROJECT, "outage", 4))
    r.check(
        local is not None and local["save_id"] != first_id,
        "outage: the save made during the outage is committed locally",
    )
    try:
        extty.delete_local_checkpoint(PROJECT, "outage", 4)
        refused = ""
    except RuntimeError as e:
        refused = str(e)
    r.check(
        "different save" in refused,
        "outage: deleting the newer local save is refused",
    )
    r.check(
        same(
            extty.load_checkpoint_from(PROJECT, "outage", 4)["model_state_dict"], second
        ),
        "outage: this machine loads its newer local save",
    )
    env.home("reader")
    r.check(
        same(
            extty.load_checkpoint_from(PROJECT, "outage", 4)["model_state_dict"], first
        ),
        "outage: another machine loads S3's save",
    )


def scenario_killed_upload(env: Env) -> None:
    import extty
    from extty.checkpoints import new_staging_dir

    r = env.results
    home = env.home("trainer")
    run = extty.Run(PROJECT, name="killed", system_metrics=False)
    small, _ = model_state(seed=7)
    run.save_checkpoint(5, state_dict=small)
    run.finish()
    first_id = env.remote_meta("killed", 5)["save_id"]

    child = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "big-save",
            "--run",
            "killed",
            "--step",
            "5",
            "--mb",
            "768",
        ],
        env={**os.environ, "EXTTY_HOME": str(home)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert child.stdout is not None
    saw_upload = False
    for line in child.stdout:
        if "uploading" in line:
            saw_upload = True
            break
    time.sleep(0.3)
    multipart = (
        env.s3()
        .list_multipart_uploads(Bucket=env.bucket, Prefix=env.step_prefix("killed", 5))
        .get("Uploads", [])
    )
    child.send_signal(signal.SIGKILL)
    child.wait()
    r.check(
        saw_upload, "killed upload: the child reached the upload before being killed"
    )
    print(f"      (multipart uploads in flight at kill: {len(multipart)})", flush=True)

    r.check(
        env.remote_meta("killed", 5)["save_id"] == first_id,
        "killed upload: S3 still names the previous save",
    )
    env.home("reader")
    r.check(
        same(
            extty.load_checkpoint_from(PROJECT, "killed", 5)["model_state_dict"], small
        ),
        "killed upload: loads return the previous save",
    )

    leftovers = staging_entries(home, "killed")
    r.check(
        [p.name.split("#")[:2] for p in leftovers]
        == [[socket.gethostname(), str(child.pid)]],
        "killed upload: the dead process left exactly one staging dir",
    )
    long_ago = time.time() - 2 * 3600
    for p in leftovers:
        os.utime(p, (long_ago, long_ago))
    shutil.rmtree(new_staging_dir(home / "runs" / PROJECT / "killed"))
    r.check(
        staging_entries(home, "killed") == [],
        "killed upload: the next save or load removes the abandoned staging dir",
    )


def scenario_distributed_load(env: Env) -> None:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_meta

    r = env.results
    env.home("trainer")
    run = extty.Run(PROJECT, name="ddp", system_metrics=False)
    model, opt = model_state(seed=8, size=4096)
    run.save_checkpoint(8, state_dict=model, optimizer_state_dict=opt)
    run.finish()
    save_id = env.remote_meta("ddp", 8)["save_id"]
    expected = digest(model)

    for trial in range(5):
        home = env.home(f"ddp-{trial}")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                __file__,
                "rank-load",
                "--run",
                "ddp",
                "--step",
                "8",
                "--expect",
                expected,
            ],
            env={**os.environ, "EXTTY_HOME": str(home)},
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0 and proc.stdout.count("RANK-OK") == 4
        if not ok:
            print(proc.stdout[-2000:], proc.stderr[-4000:], sep="\n")
        local_dir = local_checkpoint_dir(PROJECT, "ddp", 8)
        committed = (read_meta(local_dir) or {}).get("save_id") == save_id and all(
            (local_dir / n).exists() for n in ("model.pt", "optimizer.pt")
        )
        r.check(
            ok and committed and staging_entries(home, "ddp") == [],
            f"distributed load trial {trial + 1}: 4 ranks load one shared copy",
        )


def scenario_main_compat(env: Env) -> dict[str, Any]:
    import extty
    from extty.checkpoints import local_checkpoint_dir, read_meta

    r = env.results
    trainer = env.home("legacy-trainer")
    saved = env.child("main-save", "--run", "old-run", home=trainer, sdk=env.main_sdk)
    r.check(saved.returncode == 0, "main SDK: saves steps 1 and 2 to S3")
    if saved.returncode != 0:
        print(saved.stdout[-2000:], saved.stderr[-4000:], sep="\n")
        return {}
    r.check(
        "save_id" not in env.remote_meta("old-run", 1)
        and "model.pt" in env.step_keys("old-run", 1),
        "main SDK: its saves use the flat layout with no save_id",
    )

    cache_home = env.home("legacy-cache")
    cached = env.child(
        "main-load",
        "--run",
        "old-run",
        "--step",
        "1",
        home=cache_home,
        sdk=env.main_sdk,
    )
    cache_dir = local_checkpoint_dir(PROJECT, "old-run", 1)
    print(
        f"      (main SDK download left meta.json: {(cache_dir / 'meta.json').exists()})",
        flush=True,
    )
    r.check(cached.returncode == 0 and cache_dir.exists(), "main SDK: downloads step 1")

    model_1, opt_1 = model_state(seed=11)
    loaded = extty.load_checkpoint_from(PROJECT, "old-run", 1)
    r.check(
        same(loaded["model_state_dict"], model_1)
        and same(loaded["optimizer_state_dict"], opt_1),
        "new SDK: loads a main-era save over main's download cache",
    )
    r.check(
        read_meta(cache_dir) is not None,
        "new SDK: the old cache is replaced by a committed copy",
    )
    r.check(
        extty.delete_local_checkpoint(PROJECT, "old-run", 1) is True,
        "new SDK: a copy of a main-era save is deletable",
    )

    env.home("legacy-reader")
    model_2, _ = model_state(seed=12)
    r.check(
        same(
            extty.load_checkpoint_from(PROJECT, "old-run", 2)["model_state_dict"],
            model_2,
        ),
        "new SDK: loads main-era step 2 on a fresh machine",
    )

    env.home("legacy-trainer")
    run = extty.Run(PROJECT, name="old-run", system_metrics=False)
    model_13, _ = model_state(seed=13)
    run.save_checkpoint(1, state_dict=model_13)
    run.finish()
    new_id = env.remote_meta("old-run", 1)["save_id"]
    r.check(
        env.step_keys("old-run", 1) == sorted(["meta.json", f"{new_id}/model.pt"]),
        "new SDK: re-saving a main-era step removes its flat files",
    )
    env.home("legacy-reader-2")
    r.check(
        same(
            extty.load_checkpoint_from(PROJECT, "old-run", 1)["model_state_dict"],
            model_13,
        ),
        "new SDK: loads its re-save of a main-era step",
    )
    old_reader = env.child(
        "main-load",
        "--run",
        "old-run",
        "--step",
        "1",
        home=env.home_path("legacy-main-reader"),
        sdk=env.main_sdk,
    )
    print(
        "      (main SDK loading a new-layout save: "
        f"{'works' if old_reader.returncode == 0 else 'fails, as expected'})",
        flush=True,
    )
    return {"legacy_run": "old-run"}


def build_prune_fixture(env: Env) -> dict[str, Any]:
    """A home holding one local checkpoint in each state, for the TUI checks."""
    import extty

    prune = env.home("prune")
    extty.load_checkpoint_from(PROJECT, "round-trip", 1)
    main_cache = env.child(
        "main-load", "--run", "old-run", "--step", "2", home=prune, sdk=env.main_sdk
    )
    trainer_runs = env.home_path("trainer") / "runs" / PROJECT
    for run in ("no-bucket", "outage"):
        shutil.copytree(
            trainer_runs / run, prune / "runs" / PROJECT / run, dirs_exist_ok=True
        )
    env.results.check(main_cache.returncode == 0, "prune fixture: built")
    return {
        "home": str(prune),
        "deletable": [["round-trip", 1], ["old-run", 2]],
        "kept": [["no-bucket", 3], ["outage", 4]],
    }


def home_snapshot() -> tuple[bool, float]:
    """Whether the real ``~/.extty`` exists, and when it last changed."""
    real = Path.home() / ".extty"
    return real.exists(), real.stat().st_mtime if real.exists() else 0.0


def cmd_run(args: argparse.Namespace) -> int:
    real_home_before = home_snapshot()
    prefix = f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    configure_s3_env(args.endpoint, args.bucket, prefix)
    workdir = Path(args.workdir).resolve()
    shutil.rmtree(workdir / "homes", ignore_errors=True)
    env = Env(
        workdir=workdir,
        endpoint=args.endpoint,
        bucket=args.bucket,
        prefix=prefix,
        container=args.container,
        main_sdk=Path(args.main_sdk).resolve(),
        results=Results(),
    )
    print(f"S3 prefix: {prefix}", flush=True)
    for scenario in (
        scenario_round_trip,
        scenario_resave,
        scenario_failed_upload,
        scenario_outage,
        scenario_killed_upload,
        scenario_distributed_load,
    ):
        print(f"--- {scenario.__name__.removeprefix('scenario_')}", flush=True)
        scenario(env)
    print("--- main_compat", flush=True)
    scenario_main_compat(env)
    print("--- prune_fixture", flush=True)
    prune = build_prune_fixture(env)
    env.results.check(
        home_snapshot() == real_home_before,
        "isolation: this machine's own ~/.extty was not touched",
    )

    summary = {
        "prefix": prefix,
        "bucket": args.bucket,
        "project": PROJECT,
        "prune": prune,
        "round_trip_index": env.remote_index("round-trip").decode(),
        "passed": env.results.passed,
        "failed": env.results.failed,
    }
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{env.results.passed} passed, {len(env.results.failed)} failed")
    return 1 if env.results.failed else 0


def cmd_big_save(args: argparse.Namespace) -> int:
    import extty

    logging.getLogger("extty").setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.flush = sys.stdout.flush
    logging.getLogger("extty").addHandler(handler)
    run = extty.Run(PROJECT, name=args.run, system_metrics=False)
    big = {"w": torch.randn(args.mb * 1024 * 1024 // 4)}
    run.save_checkpoint(args.step, state_dict=big)
    run.finish()
    return 0


def cmd_rank_load(args: argparse.Namespace) -> int:
    import extty

    rank = os.environ.get("RANK", "?")
    loaded = extty.load_checkpoint_from(PROJECT, args.run, args.step)
    ok = (
        digest(loaded["model_state_dict"]) == args.expect
        and "optimizer_state_dict" in loaded
    )
    print(f"{'RANK-OK' if ok else 'RANK-BAD'} {rank}", flush=True)
    return 0 if ok else 1


def cmd_main_save(args: argparse.Namespace) -> int:
    import extty

    extty.init(PROJECT, name=args.run, system_metrics=False)
    for step, seed in ((1, 11), (2, 12)):
        model, opt = model_state(seed=seed)
        extty.save_checkpoint(step, state_dict=model, optimizer_state_dict=opt)
    extty.finish()
    return 0


def cmd_main_load(args: argparse.Namespace) -> int:
    import extty

    extty.load_checkpoint_from(PROJECT, args.run, args.step)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--workdir", required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--bucket", required=True)
    run.add_argument("--container", required=True)
    run.add_argument("--main-sdk", required=True)
    big = sub.add_parser("big-save")
    big.add_argument("--run", required=True)
    big.add_argument("--step", type=int, required=True)
    big.add_argument("--mb", type=int, required=True)
    rank = sub.add_parser("rank-load")
    rank.add_argument("--run", required=True)
    rank.add_argument("--step", type=int, required=True)
    rank.add_argument("--expect", required=True)
    main_save = sub.add_parser("main-save")
    main_save.add_argument("--run", required=True)
    main_load = sub.add_parser("main-load")
    main_load.add_argument("--run", required=True)
    main_load.add_argument("--step", type=int, required=True)

    args = parser.parse_args()
    commands = {
        "run": cmd_run,
        "big-save": cmd_big_save,
        "rank-load": cmd_rank_load,
        "main-save": cmd_main_save,
        "main-load": cmd_main_load,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
