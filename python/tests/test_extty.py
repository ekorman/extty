"""Tests for extty library."""

import shutil
import tempfile
from collections.abc import Generator
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
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
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
