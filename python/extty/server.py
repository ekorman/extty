"""In-memory HTTP server for remote TUI access."""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from multiprocessing import Pipe, Process, Queue
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 0
    token: str | None = None
    max_metric_points: int = 10_000
    max_example_points: int = 5_000
    max_system_points: int = 2_000

    def resolved_token(self) -> str:
        return self.token or secrets.token_hex(16)


@dataclass(frozen=True)
class ServerInfo:
    host: str
    port: int
    token: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class MetricPoint:
    step: int
    timestamp: float
    value: float


@dataclass(frozen=True)
class SystemPoint:
    step: int
    timestamp: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_mem_used_gb: float | None
    gpu_mem_total_gb: float | None
    gpu_util_pct: float | None


@dataclass(frozen=True)
class ExampleRecord:
    step: int
    timestamp: float
    data: dict[str, Any]


@dataclass
class RunBuffer:
    name: str
    project: str
    config: dict[str, Any]
    started_at: str
    status: str = "running"
    finished_at: str | None = None
    metrics: dict[str, list[MetricPoint]] = field(default_factory=dict)
    examples: dict[str, list[ExampleRecord]] = field(default_factory=dict)
    system: list[SystemPoint] = field(default_factory=list)


@dataclass(frozen=True)
class RunSummary:
    name: str
    project: str
    config: dict[str, Any]
    started_at: str
    finished_at: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "project": self.project,
            "config": self.config,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class RunStarted:
    name: str
    project: str
    config: dict[str, Any]
    started_at: str


@dataclass(frozen=True)
class RunFinished:
    name: str
    finished_at: str
    status: str


@dataclass(frozen=True)
class MetricLogged:
    name: str
    metric_name: str
    point: MetricPoint


@dataclass(frozen=True)
class ExampleLogged:
    name: str
    example_name: str
    record: ExampleRecord


@dataclass(frozen=True)
class SystemLogged:
    name: str
    point: SystemPoint


@dataclass(frozen=True)
class ShutdownServer:
    pass


ServerEvent = (
    RunStarted
    | RunFinished
    | MetricLogged
    | ExampleLogged
    | SystemLogged
    | ShutdownServer
)


@dataclass
class ServerState:
    max_metric_points: int
    max_example_points: int
    max_system_points: int
    runs: dict[str, RunBuffer] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def apply(self, event: ServerEvent) -> None:
        with self.lock:
            if isinstance(event, RunStarted):
                self.runs[event.name] = RunBuffer(
                    name=event.name,
                    project=event.project,
                    config=event.config,
                    started_at=event.started_at,
                )
            elif isinstance(event, RunFinished):
                run = self.runs.get(event.name)
                if run is not None:
                    run.finished_at = event.finished_at
                    run.status = event.status
            elif isinstance(event, MetricLogged):
                run = self._ensure_run(event.name)
                points = run.metrics.setdefault(event.metric_name, [])
                points.append(event.point)
                if len(points) > self.max_metric_points:
                    del points[: len(points) - self.max_metric_points]
            elif isinstance(event, ExampleLogged):
                run = self._ensure_run(event.name)
                records = run.examples.setdefault(event.example_name, [])
                records.append(event.record)
                if len(records) > self.max_example_points:
                    del records[: len(records) - self.max_example_points]
            elif isinstance(event, SystemLogged):
                run = self._ensure_run(event.name)
                run.system.append(event.point)
                if len(run.system) > self.max_system_points:
                    del run.system[: len(run.system) - self.max_system_points]

    def list_runs(self) -> list[str]:
        with self.lock:
            return sorted(self.runs.keys())

    def list_run_summaries(self) -> list[RunSummary]:
        with self.lock:
            return [
                RunSummary(
                    name=run.name,
                    project=run.project,
                    config=run.config,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    status=run.status,
                )
                for run in sorted(self.runs.values(), key=lambda item: item.name)
            ]

    def metrics_since(self, run_name: str, step: int) -> dict[str, list[MetricPoint]]:
        with self.lock:
            run = self.runs.get(run_name)
            if run is None:
                return {}
            return {
                name: [point for point in points if point.step > step]
                for name, points in run.metrics.items()
            }

    def examples_since(
        self, run_name: str, step: int
    ) -> dict[str, list[ExampleRecord]]:
        with self.lock:
            run = self.runs.get(run_name)
            if run is None:
                return {}
            return {
                name: [record for record in records if record.step > step]
                for name, records in run.examples.items()
            }

    def system_since(self, run_name: str, step: int) -> list[SystemPoint]:
        with self.lock:
            run = self.runs.get(run_name)
            if run is None:
                return []
            return [point for point in run.system if point.step > step]

    def _ensure_run(self, run_name: str) -> RunBuffer:
        if run_name not in self.runs:
            self.runs[run_name] = RunBuffer(
                name=run_name,
                project="unknown",
                config={},
                started_at="",
            )
        return self.runs[run_name]


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _unauthorized(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.UNAUTHORIZED)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(b'{"error":"unauthorized"}')


def _not_found(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.NOT_FOUND)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(b'{"error":"not found"}')


def _parse_step(query: dict[str, list[str]]) -> int:
    raw = query.get("step", ["0"])[0]
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _make_handler(
    state: ServerState, token: str
) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                _unauthorized(self)
                return

            parsed = urlparse(self.path)
            segments = [seg for seg in parsed.path.split("/") if seg]
            query = parse_qs(parsed.query)

            if segments == ["runs"]:
                runs = state.list_run_summaries()
                _json_response(self, {"runs": [summary.to_dict() for summary in runs]})
                return

            if len(segments) == 3 and segments[0] == "runs":
                run_name = segments[1]
                kind = segments[2]
                step = _parse_step(query)
                if kind == "metrics":
                    metrics = state.metrics_since(run_name, step)
                    payload = {
                        "metrics": [
                            {
                                "name": name,
                                "points": [
                                    {
                                        "step": point.step,
                                        "timestamp": point.timestamp,
                                        "value": point.value,
                                    }
                                    for point in points
                                ],
                            }
                            for name, points in metrics.items()
                        ]
                    }
                    _json_response(self, payload)
                    return
                if kind == "system":
                    points = state.system_since(run_name, step)
                    payload = {
                        "points": [
                            {
                                "step": point.step,
                                "timestamp": point.timestamp,
                                "ram_used_gb": point.ram_used_gb,
                                "ram_total_gb": point.ram_total_gb,
                                "gpu_mem_used_gb": point.gpu_mem_used_gb,
                                "gpu_mem_total_gb": point.gpu_mem_total_gb,
                                "gpu_util_pct": point.gpu_util_pct,
                            }
                            for point in points
                        ]
                    }
                    _json_response(self, payload)
                    return
                if kind == "examples":
                    examples = state.examples_since(run_name, step)
                    payload = {
                        "examples": [
                            {
                                "name": name,
                                "records": [
                                    {
                                        "step": record.step,
                                        "timestamp": record.timestamp,
                                        "data": record.data,
                                    }
                                    for record in records
                                ],
                            }
                            for name, records in examples.items()
                        ]
                    }
                    _json_response(self, payload)
                    return

            _not_found(self)

        def log_message(self, *_args: Any) -> None:
            return

        def _authorized(self) -> bool:
            auth_header = self.headers.get("Authorization")
            if not auth_header:
                return False
            expected = f"Bearer {token}"
            return auth_header.strip() == expected

    return Handler


def _run_server(
    settings: ServerSettings,
    event_queue: Queue[ServerEvent],
    ready_conn: Any,
) -> None:
    token = settings.resolved_token()
    state = ServerState(
        max_metric_points=settings.max_metric_points,
        max_example_points=settings.max_example_points,
        max_system_points=settings.max_system_points,
    )
    handler = _make_handler(state, token)
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    port = server.server_address[1]
    ready_conn.send((port, token))
    ready_conn.close()

    stop_event = threading.Event()

    def consume_events() -> None:
        while not stop_event.is_set():
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(event, ShutdownServer):
                stop_event.set()
                server.shutdown()
                return
            state.apply(event)

    thread = threading.Thread(target=consume_events, daemon=True)
    thread.start()

    def _check_inside_modal_fn():
        import modal

        return modal.current_function_call_id() is not None

    if find_spec("modal") is not None and _check_inside_modal_fn():
        import modal

        with modal.forward(port) as tunnel:
            print(
                f"Serving extty through modal tunnel: {tunnel.host}:{tunnel.port} with token {token}"
            )
            server.serve_forever()
    else:
        server.serve_forever()
    stop_event.set()
    thread.join(timeout=1.0)
    server.server_close()


class ServerManager:
    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._queue: Queue[ServerEvent] = Queue()
        self._process: Process | None = None
        self._info: ServerInfo | None = None

    @property
    def info(self) -> ServerInfo:
        if self._info is None:
            raise RuntimeError("Server has not been started.")
        return self._info

    @property
    def queue(self) -> Queue[ServerEvent]:
        return self._queue

    def start(self) -> None:
        if self._process is not None:
            return
        parent_conn, child_conn = Pipe(duplex=False)
        process = Process(
            target=_run_server,
            args=(self._settings, self._queue, child_conn),
            daemon=True,
        )
        process.start()
        port, token = parent_conn.recv()
        parent_conn.close()
        self._process = process
        self._info = ServerInfo(host=self._settings.host, port=port, token=token)

    def stop(self) -> None:
        if self._process is None:
            return
        self._queue.put(ShutdownServer())
        self._process.join(timeout=2.0)
        self._process = None


class QueueStorage:
    def __init__(
        self,
        manager: ServerManager,
        *,
        run_name: str,
        project: str,
        config: dict[str, Any],
        started_at: str,
    ) -> None:
        self._manager = manager
        self._run_name = run_name
        self._system_step = 0
        self._manager.queue.put(
            RunStarted(
                name=run_name,
                project=project,
                config=config,
                started_at=started_at,
            )
        )

    def log_metric(self, name: str, value: float, step: int) -> None:
        point = MetricPoint(step=step, timestamp=_now_timestamp(), value=value)
        self._manager.queue.put(
            MetricLogged(name=self._run_name, metric_name=name, point=point)
        )

    def log_example(self, name: str, data: dict[str, Any], step: int) -> None:
        record = ExampleRecord(step=step, timestamp=_now_timestamp(), data=data)
        self._manager.queue.put(
            ExampleLogged(name=self._run_name, example_name=name, record=record)
        )

    def log_system(
        self,
        ram_used_gb: float,
        ram_total_gb: float,
        gpu_mem_used_gb: float | None = None,
        gpu_mem_total_gb: float | None = None,
        gpu_util_pct: float | None = None,
    ) -> None:
        self._system_step += 1
        point = SystemPoint(
            step=self._system_step,
            timestamp=_now_timestamp(),
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            gpu_mem_used_gb=gpu_mem_used_gb,
            gpu_mem_total_gb=gpu_mem_total_gb,
            gpu_util_pct=gpu_util_pct,
        )
        self._manager.queue.put(SystemLogged(name=self._run_name, point=point))

    def flush(self) -> None:
        return

    def close(self) -> None:
        return

    def finish(self, finished_at: str, status: str) -> None:
        self._manager.queue.put(
            RunFinished(name=self._run_name, finished_at=finished_at, status=status)
        )


def _now_timestamp() -> float:
    return time.time()
