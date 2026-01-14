"""Tests for extty library."""

import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path
from unittest import mock

import pytest

import extty
from extty.storage import (
    MetaData,
    RunStorage,
    generate_run_name,
    sanitize_metric_name,
)


class TestVersion:
    def test_version_is_set(self) -> None:
        """Test that __version__ is set and is a string."""
        assert hasattr(extty, "__version__")
        assert isinstance(extty.__version__, str)
        assert len(extty.__version__) > 0

    def test_version_format(self) -> None:
        """Test that version follows expected format (either x.y.z or x.y.devN+...)."""
        version = extty.__version__
        # Version should be either a release version (e.g., "0.1.0") or
        # development version (e.g., "0.1.dev2+g8713e8cb9.d20260112")
        assert version != "0.0.0+unknown", "Version should be properly detected"
        # Should start with a digit
        assert version[0].isdigit(), f"Version should start with a digit: {version}"


class TestSanitizeMetricName:
    def test_slashes_preserved(self) -> None:
        assert sanitize_metric_name("train/loss") == "train/loss"

    def test_nested_slashes(self) -> None:
        assert sanitize_metric_name("train/metrics/loss") == "train/metrics/loss"

    def test_special_chars_in_components(self) -> None:
        assert sanitize_metric_name("val/f1@epoch") == "val/f1_epoch"

    def test_preserves_valid_chars(self) -> None:
        assert sanitize_metric_name("loss_v2.0") == "loss_v2.0"


class TestGenerateRunName:
    def test_format(self) -> None:
        name = generate_run_name()
        parts = name.split("_")
        assert len(parts) == 3
        assert len(parts[-1]) == 4


class TestRunStorage:
    @pytest.fixture
    def temp_run_dir(self) -> Generator[Path, None, None]:
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_creates_directory_structure(self, temp_run_dir: Path) -> None:
        RunStorage(run_dir=temp_run_dir / "test-run")
        assert (temp_run_dir / "test-run").exists()
        assert (temp_run_dir / "test-run" / "metrics").exists()
        assert (temp_run_dir / "test-run" / "examples").exists()

    def test_write_and_read_meta(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        meta = MetaData(
            project="test-project",
            run_name="test-run",
            config={"lr": 0.001},
            started_at="2024-01-15T14:32:00Z",
        )
        storage.write_meta(meta)

        read_meta = storage.read_meta()
        assert read_meta is not None
        assert read_meta.project == "test-project"
        assert read_meta.config == {"lr": 0.001}

    def test_log_metric_creates_csv(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        storage.log_metric("train/loss", 0.5, step=0)
        storage.flush()

        csv_path = temp_run_dir / "test-run" / "metrics" / "train" / "loss.csv"
        assert csv_path.exists()

        content = csv_path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "step,timestamp,value"
        assert lines[1].startswith("0,")
        assert lines[1].endswith(",0.5")

    def test_log_system_creates_csv(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        storage.log_system(
            ram_used_gb=8.0,
            ram_total_gb=32.0,
            gpu_mem_used_gb=4.0,
            gpu_mem_total_gb=24.0,
            gpu_util_pct=50.0,
        )

        csv_path = temp_run_dir / "test-run" / "system.csv"
        assert csv_path.exists()

        content = csv_path.read_text()
        lines = content.strip().split("\n")
        assert "ram_used_gb" in lines[0]
        assert "8.00" in lines[1]

    def test_log_example_creates_jsonl(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        storage.log_example(
            "val/example",
            {"prompt": "Hello", "response": "Hi"},
            step=10,
        )

        jsonl_path = temp_run_dir / "test-run" / "examples" / "val" / "example.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert '"prompt": "Hello"' in lines[0]


class TestExttyAPI:
    def test_init_creates_run(self, tmp_path: Path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            run = extty.init("test-project", name="my-run", system_metrics=False)
            assert run.name == "my-run"
            assert (tmp_path / "runs" / "my-run" / "meta.json").exists()
            extty.finish()

    def test_log_writes_metrics(self, tmp_path: Path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            extty.init("test-project", name="log-test", system_metrics=False)
            extty.log({"loss": 0.5, "acc": 0.8}, step=0)
            extty.log({"loss": 0.3, "acc": 0.9}, step=1)
            extty.finish()

            loss_csv = tmp_path / "runs" / "log-test" / "metrics" / "loss.csv"
            assert loss_csv.exists()
            lines = loss_csv.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_log_examples(self, tmp_path: Path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            extty.init("test-project", name="example-test", system_metrics=False)
            extty.log(
                {"val/example": {"prompt": "Capital of France?", "response": "Paris"}},
                step=10,
            )
            extty.finish()

            example_path = (
                tmp_path
                / "runs"
                / "example-test"
                / "examples"
                / "val"
                / "example.jsonl"
            )
            assert example_path.exists()
            content = example_path.read_text()
            assert '"response": "Paris"' in content

    def test_context_manager(self, tmp_path: Path) -> None:
        import json

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            with extty.init("test-project", name="ctx-test", system_metrics=False):
                extty.log({"loss": 0.5}, step=0)

            meta_path = tmp_path / "runs" / "ctx-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["status"] == "completed"

    def test_log_without_init_raises(self) -> None:
        extty._active_run = None
        with pytest.raises(RuntimeError, match="No active run"):
            extty.log({"loss": 0.5}, step=0)


class TestExttyServerMode:
    def _get_json(self, url: str, token: str) -> dict[str, object]:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))

    def _wait_for(
        self,
        fetcher: Callable[[], dict[str, object]],
        predicate: Callable[[dict[str, object]], bool],
    ) -> dict[str, object]:
        deadline = time.time() + 6.0
        last_payload: dict[str, object] = {}
        while time.time() < deadline:
            last_payload = fetcher()
            if predicate(last_payload):
                return last_payload
            time.sleep(0.1)
        return last_payload

    def _get_status(self, url: str, token: str | None) -> int:
        headers = {}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_server_endpoints_receive_logs(self) -> None:
        run = extty.init(
            "server-project",
            name="server-run",
            system_metrics=True,
            server=True,
            server_host="127.0.0.1",
        )
        assert run.server_info is not None
        token = run.server_info.token
        base_url = run.server_info.base_url

        try:
            assert self._get_status(f"{base_url}/runs", None) == 401
            assert self._get_status(f"{base_url}/runs", "bad-token") == 401

            extty.log({"loss": 0.5, "acc": 0.8}, step=1)
            extty.log({"val/example": {"prompt": "Hello", "response": "Hi"}}, step=2)
            extty.log({"loss": 0.4}, step=3)

            runs_payload = self._wait_for(
                lambda: self._get_json(f"{base_url}/runs", token),
                lambda payload: any(
                    run_data["name"] == "server-run"
                    for run_data in payload.get("runs", [])
                ),
            )
            assert any(
                run_data["name"] == "server-run" for run_data in runs_payload["runs"]
            )

            metrics_payload = self._wait_for(
                lambda: self._get_json(
                    f"{base_url}/runs/server-run/metrics?step=0", token
                ),
                lambda payload: any(
                    metric["name"] == "loss" and len(metric["points"]) == 2
                    for metric in payload.get("metrics", [])
                ),
            )
            metrics_by_name = {
                metric["name"]: metric for metric in metrics_payload["metrics"]
            }
            assert metrics_by_name["loss"]["points"][0]["value"] == 0.5
            assert metrics_by_name["loss"]["points"][1]["value"] == 0.4
            assert metrics_by_name["acc"]["points"][0]["value"] == 0.8

            examples_payload = self._wait_for(
                lambda: self._get_json(
                    f"{base_url}/runs/server-run/examples?step=0", token
                ),
                lambda payload: any(
                    example["name"] == "val/example"
                    and len(example["records"]) == 1
                    for example in payload.get("examples", [])
                ),
            )
            example_records = {
                example["name"]: example for example in examples_payload["examples"]
            }
            assert example_records["val/example"]["records"][0]["data"]["response"] == "Hi"

            system_payload = self._wait_for(
                lambda: self._get_json(
                    f"{base_url}/runs/server-run/system?step=0", token
                ),
                lambda payload: len(payload.get("points", [])) >= 1,
            )
            assert len(system_payload["points"]) >= 1
        finally:
            extty.finish()
