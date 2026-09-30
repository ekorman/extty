"""Tests for extty library."""

import dataclasses
import io
import json
import logging
import os
import pickle
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

import extty
from extty.checkpoints import (
    DELETABLE_LOCALLY,
    META_FILE,
    LocalCopy,
    adopt_download,
    checkpoint_dir,
    checkpoint_status,
    commit_checkpoint,
    discard_staged,
    local_checkpoint_dir,
    new_staging_dir,
    publish,
    read_local_checkpoint,
    remote_relpath,
    stage_checkpoint,
)
from extty.s3 import S3Config, S3Storage
from extty.storage import (
    MetaData,
    RunStorage,
    generate_random_name,
    get_artifacts_dir,
    get_extty_home,
    get_run_dir,
    get_runs_dir,
    sanitize_metric_name,
)


@contextmanager
def _runs_dir_at(runs_dir: Path) -> Iterator[None]:
    """Point extty's runs dir at ``runs_dir``, which must be named ``runs``."""
    assert runs_dir.name == "runs"
    with mock.patch.dict(os.environ, {"EXTTY_HOME": str(runs_dir.parent)}):
        yield


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
        assert sanitize_metric_name("val/f1!epoch") == "val/f1_epoch"

    def test_at_sign_preserved(self) -> None:
        assert sanitize_metric_name("val/pass@8") == "val/pass@8"

    def test_preserves_valid_chars(self) -> None:
        assert sanitize_metric_name("loss_v2.0") == "loss_v2.0"


class TestGenerateRunName:
    def test_format(self) -> None:
        name = generate_random_name()
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
        assert (temp_run_dir / "test-run" / "confusion_matrices").exists()
        assert (temp_run_dir / "test-run" / "charts").exists()
        assert (temp_run_dir / "test-run" / "images").exists()

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
        storage.flush()

        jsonl_path = temp_run_dir / "test-run" / "examples" / "val" / "example.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert '"prompt": "Hello"' in lines[0]

    def test_log_confusion_matrix_creates_jsonl(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        storage.log_confusion_matrix(
            "eval/cm",
            extty.ConfusionMatrix(matrix=[[5, 1], [0, 6]], labels=["a", "b"]),
            step=10,
        )
        storage.flush()

        jsonl_path = (
            temp_run_dir / "test-run" / "confusion_matrices" / "eval" / "cm.jsonl"
        )
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 10
        assert record["labels"] == ["a", "b"]
        assert record["matrix"] == [[5, 1], [0, 6]]

    def test_read_confusion_matrix_roundtrip(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        for step in range(3):
            storage.log_confusion_matrix(
                "eval/cm",
                extty.ConfusionMatrix(
                    matrix=[[step, 1], [0, step + 1]], labels=["a", "b"]
                ),
                step=step,
            )
        storage.flush()

        records = storage.read_confusion_matrix("eval/cm")
        assert [r.step for r in records] == [0, 1, 2]
        assert records[2].matrix == [[2, 1], [0, 3]]
        assert records[0].labels == ["a", "b"]
        assert "eval/cm" in storage.list_confusion_matrix_names()

    def test_log_chart_creates_jsonl(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        storage.log_chart(
            "eval/roc",
            extty.Chart(
                points=[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)], axis_names=("fpr", "tpr")
            ),
            step=10,
        )
        storage.flush()

        jsonl_path = temp_run_dir / "test-run" / "charts" / "eval" / "roc.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 10
        assert record["x_axis"] == "fpr"
        assert record["y_axis"] == "tpr"
        assert record["points"] == [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]

    def test_read_chart_roundtrip(self, temp_run_dir: Path) -> None:
        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        for step in range(3):
            storage.log_chart(
                "eval/roc",
                extty.Chart(
                    points=[(float(i), float(i * step)) for i in range(3)],
                    axis_names=("x", "y"),
                ),
                step=step,
            )
        storage.flush()

        records = storage.read_chart("eval/roc")
        assert [r.step for r in records] == [0, 1, 2]
        assert records[2].points == [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
        assert records[0].x_axis == "x"
        assert records[0].y_axis == "y"
        assert "eval/roc" in storage.list_chart_names()

    def test_log_image_creates_png_and_index(self, temp_run_dir: Path) -> None:
        from PIL import Image as PILImage

        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        img = PILImage.new("RGB", (8, 4), "blue")
        storage.log_image("val/dets", extty.Image(img, caption="boxes"), step=10)
        storage.flush()

        png_path = temp_run_dir / "test-run" / "images" / "val" / "dets" / "step_10.png"
        assert png_path.exists()
        assert PILImage.open(png_path).size == (8, 4)

        index_path = temp_run_dir / "test-run" / "images" / "val" / "dets.jsonl"
        assert index_path.exists()
        record = json.loads(index_path.read_text().strip())
        assert record["step"] == 10
        assert record["file"] == "val/dets/step_10.png"
        assert record["width"] == 8
        assert record["height"] == 4
        assert record["caption"] == "boxes"

    def test_read_images_dedupes_keep_last(self, temp_run_dir: Path) -> None:
        from PIL import Image as PILImage

        storage = RunStorage(run_dir=temp_run_dir / "test-run")
        for step, color in [(0, "red"), (1, "green"), (1, "blue")]:
            storage.log_image(
                "val/dets", extty.Image(PILImage.new("RGB", (4, 4), color)), step=step
            )
        storage.flush()

        records = storage.read_images("val/dets")
        assert [r.step for r in records] == [0, 1]
        assert records[0].caption is None
        assert "val/dets" in storage.list_image_names()

        png = storage.read_image_bytes(records[1].file)
        reread = PILImage.open(io.BytesIO(png)).convert("RGB")
        assert reread.getpixel((0, 0)) == (0, 0, 255)


class TestImage:
    def test_rejects_non_pil_input(self) -> None:
        with pytest.raises(TypeError, match="PIL.Image.Image"):
            extty.Image("not-an-image")  # type: ignore[arg-type]

    def test_encodes_png_with_metadata(self) -> None:
        from PIL import Image as PILImage

        img = extty.Image(PILImage.new("RGB", (16, 9), "red"), caption="c")
        assert img.width == 16
        assert img.height == 9
        assert img.mode == "RGB"
        assert img.caption == "c"
        assert img.png_bytes.startswith(b"\x89PNG")

    def test_float_mode_converts(self) -> None:
        from PIL import Image as PILImage

        img = extty.Image(PILImage.new("F", (4, 4)))
        assert img.png_bytes.startswith(b"\x89PNG")


class TestExttyAPI:
    def test_init_creates_run(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            run = extty.init("test-project", name="my-run", system_metrics=False)
            assert run.name == "my-run"
            assert (
                tmp_path / "runs" / "test-project" / "my-run" / "meta.json"
            ).exists()
            extty.finish()

    def test_log_writes_metrics(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="log-test", system_metrics=False)
            extty.log({"loss": 0.5, "acc": 0.8}, step=0)
            extty.log({"loss": 0.3, "acc": 0.9}, step=1)
            extty.finish()

            loss_csv = (
                tmp_path / "runs" / "test-project" / "log-test" / "metrics" / "loss.csv"
            )
            assert loss_csv.exists()
            lines = loss_csv.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_log_examples(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="example-test", system_metrics=False)
            extty.log(
                {
                    "val/example": extty.Example(
                        prompt="Capital of France?", responses=["Paris"]
                    )
                },
                step=10,
            )
            extty.finish()

            example_path = (
                tmp_path
                / "runs"
                / "test-project"
                / "example-test"
                / "examples"
                / "val"
                / "example.jsonl"
            )
            assert example_path.exists()
            content = example_path.read_text()
            assert '"Paris"' in content

    def test_log_confusion_matrix(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="cm-test", system_metrics=False)
            extty.log(
                {
                    "eval/cm": extty.ConfusionMatrix(
                        matrix=[[5, 1], [0, 6]], labels=["a", "b"]
                    )
                },
                step=0,
            )
            extty.finish()

        cm_path = (
            tmp_path
            / "runs"
            / "test-project"
            / "cm-test"
            / "confusion_matrices"
            / "eval"
            / "cm.jsonl"
        )
        assert cm_path.exists()
        record = json.loads(cm_path.read_text().strip())
        assert record["labels"] == ["a", "b"]
        assert record["matrix"] == [[5, 1], [0, 6]]

    def test_log_chart(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="chart-test", system_metrics=False)
            extty.log(
                {
                    "eval/roc": extty.Chart(
                        points=[(0.0, 0.0), (1.0, 1.0)], axis_names=("fpr", "tpr")
                    )
                },
                step=0,
            )
            extty.finish()

        chart_path = (
            tmp_path
            / "runs"
            / "test-project"
            / "chart-test"
            / "charts"
            / "eval"
            / "roc.jsonl"
        )
        assert chart_path.exists()
        record = json.loads(chart_path.read_text().strip())
        assert record["x_axis"] == "fpr"
        assert record["y_axis"] == "tpr"
        assert record["points"] == [[0.0, 0.0], [1.0, 1.0]]

    def test_log_image(self, tmp_path: Path) -> None:
        from PIL import Image as PILImage

        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="image-test", system_metrics=False)
            extty.log(
                {"val/dets": extty.Image(PILImage.new("RGB", (6, 3), "green"))},
                step=5,
            )
            extty.finish()

        images_dir = tmp_path / "runs" / "test-project" / "image-test" / "images"
        assert (images_dir / "val" / "dets" / "step_5.png").exists()
        record = json.loads((images_dir / "val" / "dets.jsonl").read_text().strip())
        assert record["step"] == 5
        assert record["width"] == 6
        assert "caption" not in record

        with _runs_dir_at(tmp_path / "runs"):
            run = extty.get_run("test-project", "image-test", local_only=True)
            assert run.image_names == ["val/dets"]
            records = run.images("val/dets")
            assert [r.step for r in records] == [5]
            assert run.image_bytes(records[0]).startswith(b"\x89PNG")

    def test_context_manager(self, tmp_path: Path) -> None:
        import json

        with _runs_dir_at(tmp_path / "runs"):
            with extty.init("test-project", name="ctx-test", system_metrics=False):
                extty.log({"loss": 0.5}, step=0)

            meta_path = tmp_path / "runs" / "test-project" / "ctx-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["status"] == "completed"

    def test_log_without_init_raises(self) -> None:
        extty._active_run = None
        with pytest.raises(RuntimeError, match="No active run"):
            extty.log({"loss": 0.5}, step=0)


class TestInstanceId:
    def test_instance_id_from_env(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            env = {"EXTTY_INSTANCE_ID": "i-abc123", "EXTTY_INSTANCE_PROVIDER": "lambda"}
            with mock.patch.dict(os.environ, env):
                extty.init("test-project", name="inst-test", system_metrics=False)
                extty.finish()

            meta_path = tmp_path / "runs" / "test-project" / "inst-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["_instance_id"] == "lambda:i-abc123"

    def test_instance_id_missing_when_no_env(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            env_remove = {
                k: ""
                for k in ("EXTTY_INSTANCE_ID", "EXTTY_INSTANCE_PROVIDER")
                if k in os.environ
            }
            with mock.patch.dict(os.environ, env_remove, clear=False):
                for k in ("EXTTY_INSTANCE_ID", "EXTTY_INSTANCE_PROVIDER"):
                    os.environ.pop(k, None)
                extty.init("test-project", name="no-inst-test", system_metrics=False)
                extty.finish()

            meta_path = (
                tmp_path / "runs" / "test-project" / "no-inst-test" / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert "_instance_id" not in meta["config"]

    def test_instance_id_missing_when_partial_env(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            env = {"EXTTY_INSTANCE_ID": "i-abc123"}
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop("EXTTY_INSTANCE_PROVIDER", None)
                extty.init(
                    "test-project", name="partial-inst-test", system_metrics=False
                )
                extty.finish()

            meta_path = (
                tmp_path / "runs" / "test-project" / "partial-inst-test" / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert "_instance_id" not in meta["config"]


class TestConfusionMatrix:
    """Tests for ConfusionMatrix construction and validation."""

    def test_basic(self) -> None:
        cm = extty.ConfusionMatrix(matrix=[[5, 1], [0, 6]], labels=["a", "b"])
        assert cm.matrix == [[5, 1], [0, 6]]
        assert cm.labels == ["a", "b"]

    def test_row_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matrix rows"):
            extty.ConfusionMatrix(matrix=[[1, 0]], labels=["a", "b"])

    def test_row_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matrix row"):
            extty.ConfusionMatrix(matrix=[[1, 0, 0], [0, 1, 0]], labels=["a", "b"])

    def test_from_array_with_tolist(self) -> None:
        class Fake:
            def tolist(self) -> list[list[int]]:
                return [[1, 2], [3, 4]]

        cm = extty.ConfusionMatrix.from_array(Fake(), ["x", "y"])
        assert cm.matrix == [[1, 2], [3, 4]]
        assert cm.labels == ["x", "y"]

    def test_from_array_with_plain_list(self) -> None:
        cm = extty.ConfusionMatrix.from_array([[0, 1], [2, 3]], ["a", "b"])
        assert cm.matrix == [[0, 1], [2, 3]]


class TestChart:
    """Tests for Chart construction and validation."""

    def test_basic(self) -> None:
        chart = extty.Chart(points=[(0.0, 1.0), (2.0, 3.0)], axis_names=("x", "y"))
        assert chart.points == [(0.0, 1.0), (2.0, 3.0)]
        assert chart.axis_names == ("x", "y")

    def test_bad_axis_names_raises(self) -> None:
        with pytest.raises(ValueError, match="axis_names"):
            extty.Chart(points=[(0.0, 1.0)], axis_names=("x",))  # type: ignore[arg-type]

    def test_bad_point_raises(self) -> None:
        with pytest.raises(ValueError, match="point"):
            extty.Chart(points=[(0.0, 1.0, 2.0)], axis_names=("x", "y"))  # type: ignore[list-item]

    def test_from_arrays_with_tolist(self) -> None:
        class Fake:
            def tolist(self) -> list[float]:
                return [1.0, 2.0, 3.0]

        chart = extty.Chart.from_arrays(Fake(), Fake(), ["x", "y"])
        assert chart.points == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        assert chart.axis_names == ("x", "y")

    def test_from_arrays_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            extty.Chart.from_arrays([1, 2], [1], ["x", "y"])


class TestExampleRewards:
    """Tests for Example and BatchExample reward functionality."""

    def test_example_with_scalar_rewards(self) -> None:
        """Test Example with scalar reward values."""
        example = extty.Example(
            prompt="What is 2+2?",
            responses=["4", "Four", "The answer is 4"],
            rewards=[0.95, 0.85, 0.90],
        )
        assert example.rewards == [0.95, 0.85, 0.90]
        result = example.to_dict()
        assert result["reward"] == [[0.95, 0.85, 0.90]]

    def test_example_with_component_rewards(self) -> None:
        """Test Example with component-based rewards (dict)."""
        example = extty.Example(
            prompt="Write a factorial function",
            responses=["def factorial(n): ...", "def fact(n): ..."],
            rewards=[
                {"acc": 1.0, "fmt": 0.9, "eff": 0.7},
                {"acc": 1.0, "fmt": 0.5, "eff": 0.95},
            ],
        )
        assert example.rewards is not None
        assert example.rewards[0] == {"acc": 1.0, "fmt": 0.9, "eff": 0.7}
        result = example.to_dict()
        assert result["reward"] == [
            [
                {"acc": 1.0, "fmt": 0.9, "eff": 0.7},
                {"acc": 1.0, "fmt": 0.5, "eff": 0.95},
            ]
        ]

    def test_example_without_rewards(self) -> None:
        """Test Example without rewards (backward compatibility)."""
        example = extty.Example(
            prompt="Hello",
            responses=["Hi", "Hello!"],
        )
        assert example.rewards is None
        result = example.to_dict()
        assert "reward" not in result

    def test_example_rewards_length_mismatch_raises(self) -> None:
        """Test that mismatched rewards/responses length raises ValueError."""
        with pytest.raises(ValueError, match="rewards length .* != responses length"):
            extty.Example(
                prompt="Test",
                responses=["A", "B", "C"],
                rewards=[0.5, 0.6],  # Only 2 rewards for 3 responses
            )

    def test_batch_example_with_scalar_rewards(self) -> None:
        """Test BatchExample with scalar rewards."""
        batch = extty.BatchExample(
            prompts=["What is 2+2?", "What is 3+3?"],
            responses=[["4", "Four"], ["6", "Six"]],
            rewards=[[0.95, 0.85], [0.90, 0.80]],
        )
        assert batch.rewards == [[0.95, 0.85], [0.90, 0.80]]
        result = batch.to_dict()
        assert result["reward"] == [[0.95, 0.85], [0.90, 0.80]]

    def test_batch_example_with_component_rewards(self) -> None:
        """Test BatchExample with component-based rewards."""
        batch = extty.BatchExample(
            prompts=["Prompt 1", "Prompt 2"],
            responses=[["R1a", "R1b"], ["R2a"]],
            rewards=[
                [{"acc": 1.0, "fmt": 0.9}, {"acc": 0.8, "fmt": 0.7}],
                [{"acc": 0.95, "fmt": 0.85}],
            ],
        )
        assert batch.rewards is not None
        assert batch.rewards[0][0] == {"acc": 1.0, "fmt": 0.9}
        result = batch.to_dict()
        assert "reward" in result

    def test_batch_example_without_rewards(self) -> None:
        """Test BatchExample without rewards (backward compatibility)."""
        batch = extty.BatchExample(
            prompts=["P1", "P2"],
            responses=[["R1"], ["R2"]],
        )
        assert batch.rewards is None
        result = batch.to_dict()
        assert "reward" not in result

    def test_batch_example_rewards_outer_length_mismatch_raises(self) -> None:
        """Test that mismatched outer rewards/prompts length raises ValueError."""
        with pytest.raises(ValueError, match="rewards length .* != prompts length"):
            extty.BatchExample(
                prompts=["P1", "P2", "P3"],
                responses=[["R1"], ["R2"], ["R3"]],
                rewards=[[0.5], [0.6]],  # Only 2 reward groups for 3 prompts
            )

    def test_batch_example_rewards_inner_length_mismatch_raises(self) -> None:
        """Test that mismatched inner rewards/responses length raises ValueError."""
        with pytest.raises(
            ValueError, match=r"rewards\[1\] length .* != responses\[1\] length"
        ):
            extty.BatchExample(
                prompts=["P1", "P2"],
                responses=[["R1a", "R1b"], ["R2a", "R2b", "R2c"]],
                rewards=[
                    [0.5, 0.6],
                    [0.7, 0.8],
                ],  # Second group has 2 rewards for 3 responses
            )

    def test_log_example_with_rewards(self, tmp_path: Path) -> None:
        """Integration test: log Example with rewards and verify JSONL output."""
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="reward-test", system_metrics=False)
            extty.log(
                {
                    "val/example": extty.Example(
                        prompt="What is 2+2?",
                        responses=["4", "Four"],
                        rewards=[0.95, 0.80],
                    )
                },
                step=10,
            )
            extty.finish()

            example_path = (
                tmp_path
                / "runs"
                / "test-project"
                / "reward-test"
                / "examples"
                / "val"
                / "example.jsonl"
            )
            assert example_path.exists()
            content = example_path.read_text()
            data = json.loads(content)
            assert data["data"]["reward"] == [[0.95, 0.80]]

    def test_log_example_with_component_rewards(self, tmp_path: Path) -> None:
        """Integration test: log Example with component rewards."""
        with _runs_dir_at(tmp_path / "runs"):
            extty.init(
                "test-project", name="component-reward-test", system_metrics=False
            )
            extty.log(
                {
                    "train/code": extty.Example(
                        prompt="Write factorial",
                        responses=["def factorial(n): ..."],
                        rewards=[{"acc": 1.0, "fmt": 0.9, "eff": 0.85}],
                    )
                },
                step=100,
            )
            extty.finish()

            example_path = (
                tmp_path
                / "runs"
                / "test-project"
                / "component-reward-test"
                / "examples"
                / "train"
                / "code.jsonl"
            )
            assert example_path.exists()
            content = example_path.read_text()
            data = json.loads(content)
            assert data["data"]["reward"] == [[{"acc": 1.0, "fmt": 0.9, "eff": 0.85}]]


class TestExampleGroundtruth:
    """Tests for Example and BatchExample groundtruth functionality."""

    def test_example_with_groundtruth(self) -> None:
        example = extty.Example(
            prompt="What is 2+2?",
            responses=["4", "Five"],
            groundtruth="4",
        )
        assert example.groundtruth == "4"
        result = example.to_dict()
        assert result["groundtruth"] == ["4"]

    def test_example_without_groundtruth(self) -> None:
        example = extty.Example(
            prompt="Hello",
            responses=["Hi"],
        )
        assert example.groundtruth is None
        result = example.to_dict()
        assert "groundtruth" not in result

    def test_batch_example_with_groundtruth(self) -> None:
        batch = extty.BatchExample(
            prompts=["What is 2+2?", "What is 3+3?"],
            responses=[["4"], ["6"]],
            groundtruth=["4", "6"],
        )
        assert batch.groundtruth == ["4", "6"]
        result = batch.to_dict()
        assert result["groundtruth"] == ["4", "6"]

    def test_batch_example_groundtruth_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="groundtruth length .* != prompts length"):
            extty.BatchExample(
                prompts=["P1", "P2", "P3"],
                responses=[["R1"], ["R2"], ["R3"]],
                groundtruth=["GT1", "GT2"],
            )

    def test_log_example_with_groundtruth(self, tmp_path: Path) -> None:
        with _runs_dir_at(tmp_path / "runs"):
            extty.init("test-project", name="gt-test", system_metrics=False)
            extty.log(
                {
                    "val/example": extty.Example(
                        prompt="What is 2+2?",
                        responses=["4"],
                        groundtruth="4",
                    )
                },
                step=10,
            )
            extty.finish()

            example_path = (
                tmp_path
                / "runs"
                / "test-project"
                / "gt-test"
                / "examples"
                / "val"
                / "example.jsonl"
            )
            assert example_path.exists()
            content = example_path.read_text()
            data = json.loads(content)
            assert data["data"]["groundtruth"] == ["4"]


class TestExperimentDecorator:
    def test_decorator_initializes_and_finishes_run(self, tmp_path: Path) -> None:
        """Test that the decorator properly initializes and finishes a run."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment(
                "test-project", name="decorator-test", system_metrics=False
            )
            def my_experiment(lr: float = 0.01, epochs: int = 10) -> str:
                return "done"

            result = my_experiment(lr=0.001, epochs=5)

            assert result == "done"
            # Verify run was finished (no active run)
            assert extty._active_run is None
            # Verify meta.json was created
            meta_path = (
                tmp_path / "runs" / "test-project" / "decorator-test" / "meta.json"
            )
            assert meta_path.exists()

    def test_decorator_logs_kwargs_as_config(self, tmp_path: Path) -> None:
        """Test that kwargs passed to the decorated function are logged as config."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="config-test", system_metrics=False)
            def my_experiment(lr: float = 0.01, batch_size: int = 32) -> None:
                pass

            my_experiment(lr=0.001, batch_size=64)

            meta_path = tmp_path / "runs" / "test-project" / "config-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["lr"] == 0.001
            assert meta["config"]["batch_size"] == 64

    def test_decorator_warns_on_positional_args(self, tmp_path: Path) -> None:
        """Test that positional arguments trigger a warning."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="args-test", system_metrics=False)
            def my_experiment(lr: float, epochs: int) -> None:
                pass

            with pytest.warns(UserWarning, match="non-keyword args"):
                my_experiment(0.01, epochs=10)

    def test_decorator_finishes_run_on_exception(self, tmp_path: Path) -> None:
        """Test that the run is finished even when an exception is raised."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment(
                "test-project", name="exception-test", system_metrics=False
            )
            def my_experiment() -> None:
                raise ValueError("Something went wrong")

            with pytest.raises(ValueError, match="Something went wrong"):
                my_experiment()

            # Verify run was still finished
            assert extty._active_run is None

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that functools.wraps preserves the original function metadata."""

        @extty.experiment("test-project", system_metrics=False)
        def my_documented_experiment(lr: float = 0.01) -> int:
            """This is my experiment docstring."""
            return 42

        assert my_documented_experiment.__name__ == "my_documented_experiment"
        assert my_documented_experiment.__doc__ == "This is my experiment docstring."

    def test_decorator_returns_correct_value(self, tmp_path: Path) -> None:
        """Test that the decorator returns the function's return value."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="return-test", system_metrics=False)
            def my_experiment() -> dict:
                return {"accuracy": 0.95, "loss": 0.05}

            result = my_experiment()

            assert result == {"accuracy": 0.95, "loss": 0.05}

    def test_decorator_name_kwarg(self, tmp_path: Path) -> None:
        """Test that name_kwarg uses a kwarg value as the run name."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment(
                "test-project", name_kwarg="run_name", system_metrics=False
            )
            def my_experiment(run_name: str, lr: float = 0.01) -> None:
                pass

            my_experiment(run_name="custom-run-name", lr=0.001)

            meta_path = (
                tmp_path / "runs" / "test-project" / "custom-run-name" / "meta.json"
            )
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["run_name"] == "custom-run-name"
            assert "run_name" not in meta["config"]
            assert meta["config"]["lr"] == 0.001

    def test_decorator_conf_kwargs(self, tmp_path: Path) -> None:
        """Test that conf_kwargs only logs specified kwargs to config."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment(
                "test-project",
                name="conf-kwargs-test",
                conf_kwargs=["lr", "batch_size"],
                system_metrics=False,
            )
            def my_experiment(
                lr: float, batch_size: int, data_path: str, verbose: bool = False
            ) -> None:
                pass

            my_experiment(lr=0.001, batch_size=64, data_path="/data", verbose=True)

            meta_path = (
                tmp_path / "runs" / "test-project" / "conf-kwargs-test" / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["lr"] == 0.001
            assert meta["config"]["batch_size"] == 64
            assert "data_path" not in meta["config"]
            assert "verbose" not in meta["config"]

    def test_decorator_non_conf_kwargs(self, tmp_path: Path) -> None:
        """Test that non_conf_kwargs excludes specified kwargs from config."""
        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment(
                "test-project",
                name="non-conf-kwargs-test",
                non_conf_kwargs=["data_path", "verbose"],
                system_metrics=False,
            )
            def my_experiment(
                lr: float, batch_size: int, data_path: str, verbose: bool = False
            ) -> None:
                pass

            my_experiment(lr=0.001, batch_size=64, data_path="/data", verbose=True)

            meta_path = (
                tmp_path
                / "runs"
                / "test-project"
                / "non-conf-kwargs-test"
                / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["lr"] == 0.001
            assert meta["config"]["batch_size"] == 64
            assert "data_path" not in meta["config"]
            assert "verbose" not in meta["config"]

    def test_decorator_conf_and_non_conf_kwargs_raises(self) -> None:
        """Test that passing both conf_kwargs and non_conf_kwargs raises ValueError."""
        with pytest.raises(
            ValueError,
            match="cannot pass values for both.*conf_kwargs.*non_conf_kwargs",
        ):

            @extty.experiment(
                "test-project",
                conf_kwargs=["lr"],
                non_conf_kwargs=["verbose"],
                system_metrics=False,
            )
            def my_experiment(lr: float, verbose: bool) -> None:
                pass


class TestExperimentDataclassConfig:
    def test_dataclass_kwarg_serialized_as_dict(self, tmp_path: Path) -> None:
        @dataclasses.dataclass
        class OptimizerConfig:
            lr: float = 0.01
            momentum: float = 0.9

        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="dc-test", system_metrics=False)
            def my_experiment(optimizer: OptimizerConfig, epochs: int = 10) -> None:
                pass

            my_experiment(optimizer=OptimizerConfig(lr=0.001, momentum=0.95), epochs=5)

            meta_path = tmp_path / "runs" / "test-project" / "dc-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["optimizer"] == {"lr": 0.001, "momentum": 0.95}
            assert meta["config"]["epochs"] == 5

    def test_nested_dataclass_in_list(self, tmp_path: Path) -> None:
        @dataclasses.dataclass
        class LayerConfig:
            units: int
            activation: str = "relu"

        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="dc-list-test", system_metrics=False)
            def my_experiment(layers: list[LayerConfig] | None = None) -> None:
                pass

            my_experiment(layers=[LayerConfig(128), LayerConfig(64, "tanh")])

            meta_path = (
                tmp_path / "runs" / "test-project" / "dc-list-test" / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["layers"] == [
                {"units": 128, "activation": "relu"},
                {"units": 64, "activation": "tanh"},
            ]

    def test_nested_dataclass_in_dict(self, tmp_path: Path) -> None:
        @dataclasses.dataclass
        class SchedulerConfig:
            step_size: int = 10
            gamma: float = 0.1

        with _runs_dir_at(tmp_path / "runs"):

            @extty.experiment("test-project", name="dc-dict-test", system_metrics=False)
            def my_experiment(
                schedulers: dict[str, SchedulerConfig] | None = None,
            ) -> None:
                pass

            my_experiment(schedulers={"warmup": SchedulerConfig(5, 0.5)})

            meta_path = (
                tmp_path / "runs" / "test-project" / "dc-dict-test" / "meta.json"
            )
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["schedulers"] == {
                "warmup": {"step_size": 5, "gamma": 0.5}
            }


_CHECKPOINT_STATUS_SPEC = json.loads(
    (Path(__file__).parents[2] / "spec" / "checkpoint_status.json").read_text()
)


def _pickle_save(obj, path):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def fake_torch() -> Generator[mock.MagicMock, None, None]:
    """Stand in for a CPU-only torch with pickle-backed ``save`` / ``load``."""
    fake = mock.MagicMock()
    fake.cuda.is_available.return_value = False
    fake.backends.mps.is_available.return_value = False
    fake.save.side_effect = _pickle_save
    fake.load.side_effect = lambda path, **kw: _pickle_load(path)
    with mock.patch.dict("sys.modules", {"torch": fake}):
        yield fake


def _mock_s3_client() -> tuple[mock.MagicMock, dict[str, bytes]]:
    """A mock S3 client backed by an in-memory ``{key: bytes}`` store."""
    stored: dict[str, bytes] = {}
    client = mock.MagicMock()
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

    def require(key: str) -> bytes:
        if key not in stored:
            raise client.exceptions.NoSuchKey({"Error": {"Code": "NoSuchKey"}}, key)
        return stored[key]

    def put_object(Bucket, Key, Body, ContentType=None):
        stored[Key] = Body

    def get_object(Bucket, Key):
        body = mock.MagicMock()
        body.read.return_value = require(Key)
        return {"Body": body}

    def head_object(Bucket, Key):
        return {"ContentLength": len(require(Key))}

    def upload_file(Filename, Bucket, Key, **kwargs):
        stored[Key] = Path(Filename).read_bytes()

    def download_file(Bucket, Key, Filename, **kwargs):
        data = require(Key)
        Path(Filename).write_bytes(data)
        if kwargs.get("Callback") is not None:
            kwargs["Callback"](len(data))

    def paginate(Bucket, Prefix):
        keys = sorted(k for k in stored if k.startswith(Prefix))
        return [{"Contents": [{"Key": k} for k in keys]}]

    def delete_objects(Bucket, Delete):
        for obj in Delete["Objects"]:
            stored.pop(obj["Key"], None)

    client.put_object.side_effect = put_object
    client.get_object.side_effect = get_object
    client.head_object.side_effect = head_object
    client.upload_file.side_effect = upload_file
    client.download_file.side_effect = download_file
    client.get_paginator.return_value.paginate.side_effect = paginate
    client.delete_objects.side_effect = delete_objects
    client.delete_object.side_effect = lambda Bucket, Key: stored.pop(Key, None)
    return client, stored


_S3_CONFIG = S3Config(bucket="b", prefix="pfx")


def _s3_storage(client: mock.MagicMock, run_name: str = "run-1") -> S3Storage:
    with mock.patch("boto3.client", return_value=client):
        return S3Storage(_S3_CONFIG, "proj", run_name)


def _s3_run(client: mock.MagicMock, name: str) -> extty.run.Run:
    with mock.patch("boto3.client", return_value=client):
        return extty.run.Run(
            "proj", name=name, system_metrics=False, s3_config=_S3_CONFIG
        )


def _step_prefix(run_name: str, step: int) -> str:
    return f"pfx/runs/proj/{run_name}/checkpoints/{step}"


def _index_key(run_name: str) -> str:
    return f"pfx/runs/proj/{run_name}/checkpoints.json"


def _remote_meta(stored: dict[str, bytes], run_name: str, step: int) -> dict[str, Any]:
    return json.loads(stored[f"{_step_prefix(run_name, step)}/{META_FILE}"])


def _remote_file(
    stored: dict[str, bytes], run_name: str, step: int, name: str
) -> bytes:
    """A file of the save S3's ``meta.json`` currently points at."""
    meta = _remote_meta(stored, run_name, step)
    return stored[f"{_step_prefix(run_name, step)}/{remote_relpath(meta, name)}"]


def _step_keys(stored: dict[str, bytes], run_name: str, step: int) -> list[str]:
    prefix = _step_prefix(run_name, step) + "/"
    return sorted(k.removeprefix(prefix) for k in stored if k.startswith(prefix))


def _save_local(run_dir: Path, step: int, **kwargs: Any) -> dict[str, Any]:
    """Stage and commit a checkpoint into ``run_dir``, returning its meta."""
    staged = stage_checkpoint(run_dir, step, **kwargs)
    try:
        commit_checkpoint(run_dir, staged)
    finally:
        discard_staged(staged)
    return staged.meta


def _write_and_upload(
    storage: S3Storage, step: int, **kwargs: Any
) -> dict[str, Any] | None:
    """Stage a checkpoint for ``storage``'s run and upload it, as Run does."""
    staged = stage_checkpoint(
        get_run_dir(storage.project, storage.run_name), step, **kwargs
    )
    try:
        uploaded = storage.upload_checkpoint(staged.meta, staged.sources)
    finally:
        discard_staged(staged)
    return staged.meta if uploaded else None


def _staging_leftovers(run_dir: Path) -> list[Path]:
    staging = run_dir / "checkpoints" / ".staging"
    return list(staging.iterdir()) if staging.exists() else []


class TestCheckpointStatus:
    """The local-vs-S3 status table, shared with the TUI."""

    @pytest.mark.parametrize(
        "case",
        _CHECKPOINT_STATUS_SPEC["cases"],
        ids=[case["name"] for case in _CHECKPOINT_STATUS_SPEC["cases"]],
    )
    def test_matches_spec(self, case: dict[str, Any]) -> None:
        local = (
            None
            if case["local"] is None
            else LocalCopy(meta=None if case["local"] == "untracked" else case["local"])
        )
        status = checkpoint_status(local, case["remote"])

        assert (status.value if status else None) == case["status"]
        assert (status in DELETABLE_LOCALLY) == case["deletable"]


class TestLocalCheckpoints:
    """Staging and committing checkpoints in the run dir."""

    def test_committed_checkpoint_layout(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """State dicts become model.pt / optimizer.pt plus the save's meta.json."""
        meta = _save_local(
            tmp_path, 3, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1}
        )

        dest = checkpoint_dir(tmp_path, 3)
        assert meta["step"] == 3
        assert meta["save_id"]
        assert [f["name"] for f in meta["files"]] == ["model.pt", "optimizer.pt"]
        assert all(f["size_bytes"] > 0 for f in meta["files"])
        assert json.loads((dest / META_FILE).read_text()) == meta
        assert sorted(p.name for p in dest.iterdir()) == [
            "meta.json",
            "model.pt",
            "optimizer.pt",
        ]
        assert _staging_leftovers(tmp_path) == []

    def test_each_save_gets_its_own_id(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        with mock.patch("time.strftime", return_value="2026-09-30T10:00:00+0000"):
            first = _save_local(tmp_path, 3, state_dict={"w": 1})
            second = _save_local(tmp_path, 3, state_dict={"w": 1})

        assert first["save_id"] != second["save_id"]

    def test_stage_checkpoint_requires_exactly_one_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            stage_checkpoint(tmp_path, 1)

        with pytest.raises(ValueError, match="Exactly one"):
            stage_checkpoint(tmp_path, 1, path="/a", state_dict={"k": "v"})

    def test_staging_leaves_committed_save_untouched(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """Nothing in the step dir changes until a new save is committed."""
        _save_local(tmp_path, 3, state_dict={"w": 1})

        staged = stage_checkpoint(tmp_path, 3, state_dict={"w": 2})
        assert read_local_checkpoint(checkpoint_dir(tmp_path, 3)) == {
            "model_state_dict": {"w": 1}
        }
        discard_staged(staged)
        assert _staging_leftovers(tmp_path) == []

    def test_failed_serialization_keeps_previous_save(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        _save_local(tmp_path, 3, state_dict={"w": 1})

        def partial_save(obj: Any, path: Path) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        fake_torch.save.side_effect = partial_save
        with pytest.raises(OSError, match="disk full"):
            stage_checkpoint(tmp_path, 3, state_dict={"w": 2})

        assert read_local_checkpoint(checkpoint_dir(tmp_path, 3)) == {
            "model_state_dict": {"w": 1}
        }
        assert _staging_leftovers(tmp_path) == []

    def test_resave_replaces_whole_directory(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        _save_local(tmp_path, 3, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1})
        _save_local(tmp_path, 3, state_dict={"w": 2})

        dest = checkpoint_dir(tmp_path, 3)
        assert sorted(p.name for p in dest.iterdir()) == ["meta.json", "model.pt"]
        assert read_local_checkpoint(dest) == {"model_state_dict": {"w": 2}}
        assert _staging_leftovers(tmp_path) == []

    def test_failed_replace_restores_previous_save(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        _save_local(tmp_path, 3, state_dict={"w": 1})
        dest = checkpoint_dir(tmp_path, 3)

        with pytest.raises(FileNotFoundError):
            publish(tmp_path / "checkpoints" / ".staging" / "gone", dest)

        assert read_local_checkpoint(dest) == {"model_state_dict": {"w": 1}}

    def test_adopt_download_adds_files_to_same_save(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """A second loader of a step fills in what the first one didn't fetch."""
        meta = _save_local(
            tmp_path, 3, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1}
        )
        dest = checkpoint_dir(tmp_path, 3)
        (dest / "optimizer.pt").rename(tmp_path / "optimizer.pt")
        staging = new_staging_dir(tmp_path)
        (tmp_path / "optimizer.pt").rename(staging / "optimizer.pt")
        (staging / META_FILE).write_text(json.dumps(meta))

        assert adopt_download(staging, dest, meta) is True
        assert read_local_checkpoint(dest) == {
            "model_state_dict": {"w": 1},
            "optimizer_state_dict": {"lr": 0.1},
        }

    def test_adopt_download_leaves_a_different_save_alone(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        _save_local(tmp_path, 3, state_dict={"w": 1})
        dest = checkpoint_dir(tmp_path, 3)
        staging = new_staging_dir(tmp_path)
        other = {"step": 3, "save_id": "other", "timestamp": "t", "files": []}
        (staging / META_FILE).write_text(json.dumps(other))

        assert adopt_download(staging, dest, other) is False
        assert read_local_checkpoint(dest) == {"model_state_dict": {"w": 1}}

    def test_staging_left_by_dead_processes_is_removed(self, tmp_path: Path) -> None:
        """Only this host's staging dirs, from dead processes, left long enough."""
        exited = subprocess.Popen(["true"])
        exited.wait()
        host = socket.gethostname()
        staging_root = tmp_path / "checkpoints" / ".staging"
        names = {
            "abandoned": f"{host}#{exited.pid}#a",
            "recent": f"{host}#{exited.pid}#b",
            "running": f"{host}#{os.getpid()}#c",
            "other_host": f"elsewhere#{exited.pid}#d",
            "set_aside": f"{host}#{exited.pid}#e#replaced",
        }
        long_ago = time.time() - 2 * 3600
        for label, name in names.items():
            (staging_root / name).mkdir(parents=True)
            (staging_root / name / "model.pt").write_bytes(b"w")
            if label != "recent":
                os.utime(staging_root / name, (long_ago, long_ago))

        new_staging_dir(tmp_path)

        remaining = {p.name for p in staging_root.iterdir()}
        assert names["abandoned"] not in remaining
        assert {names[k] for k in names if k != "abandoned"} <= remaining

    def test_read_local_checkpoint_round_trip(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        _save_local(tmp_path, 3, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1})

        dest = checkpoint_dir(tmp_path, 3)
        assert read_local_checkpoint(dest) == {
            "model_state_dict": {"w": 1},
            "optimizer_state_dict": {"lr": 0.1},
        }
        assert read_local_checkpoint(dest, load_optimizer=False) == {
            "model_state_dict": {"w": 1}
        }

    def test_read_local_checkpoint_requires_meta_and_needed_files(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """A missing optimizer matters only if it is asked for."""
        dest = checkpoint_dir(tmp_path, 3)
        assert read_local_checkpoint(dest) is None

        _save_local(tmp_path, 3, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1})
        (dest / "optimizer.pt").unlink()
        assert read_local_checkpoint(dest) is None
        assert read_local_checkpoint(dest, load_optimizer=False) == {
            "model_state_dict": {"w": 1}
        }

        (dest / META_FILE).unlink()
        assert read_local_checkpoint(dest, load_optimizer=False) is None

    def test_run_saves_and_loads_without_s3(
        self,
        extty_home: Path,
        fake_torch: mock.MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no S3 configured, checkpoints live in the run dir and round-trip."""
        with caplog.at_level(logging.INFO, logger="extty"):
            run = extty.run.Run("proj", name="local-run", system_metrics=False)
            run.save_checkpoint(
                5, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1}
            )
            loaded = run.load_checkpoint(5)
            run.finish()

        expected = {"model_state_dict": {"w": 1}, "optimizer_state_dict": {"lr": 0.1}}
        run_dir = extty_home / "runs" / "proj" / "local-run"
        ckpt_dir = run_dir / "checkpoints" / "5"
        assert loaded == expected
        assert (ckpt_dir / "model.pt").exists()
        assert f"saved to {ckpt_dir}" in caplog.text
        assert not (run_dir / "checkpoints.json").exists()
        run_data = extty.get_run("proj", "local-run", local_only=True)
        assert [c.step for c in run_data.checkpoints] == [5]
        assert extty.load_checkpoint_from("proj", "local-run", 5) == expected

    def test_run_load_missing_step_without_s3_raises(
        self, fake_torch: mock.MagicMock
    ) -> None:
        run = extty.run.Run("proj", name="empty-run", system_metrics=False)
        with pytest.raises(FileNotFoundError, match="step 9"):
            run.load_checkpoint(9)
        run.finish()

    def test_load_checkpoint_from_missing_without_s3_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="S3 is not configured"):
            extty.load_checkpoint_from("proj", "nope", 1)

    def test_missing_path_leaves_existing_local_copy_intact(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """A bad path fails before touching the step's existing checkpoint."""
        run = extty.run.Run("proj", name="typo", system_metrics=False)
        run.save_checkpoint(5, state_dict={"w": 1})

        with pytest.raises(FileNotFoundError):
            run.save_checkpoint(5, path="/nonexistent/ckpt.pt")

        assert run.load_checkpoint(5) == {"model_state_dict": {"w": 1}}
        run.finish()

    def test_run_and_lookups_share_run_dir(self) -> None:
        """Where a Run saves must be where lookups by project/name look."""
        run = extty.run.Run("", name="shared", system_metrics=False)
        run.finish()

        run_dir = Path(run.run_dir)
        assert run_dir == get_run_dir("", "shared")
        assert run_dir == get_runs_dir() / "_default" / "shared"
        assert local_checkpoint_dir("", "shared", 3) == run_dir / "checkpoints" / "3"

    def test_module_level_save_checkpoint_without_init_raises(self) -> None:
        extty._active_run = None
        with pytest.raises(RuntimeError, match="No active run"):
            extty.save_checkpoint(step=1, path="/nonexistent")


class TestLocalCheckpointIndex:
    """``checkpoints.json`` mirrors S3's index; local saves are found on disk."""

    def test_record_checkpoint_merges_and_sorts(self, tmp_path: Path) -> None:
        storage = RunStorage(run_dir=tmp_path / "run")

        storage.record_checkpoint(
            {"step": 100, "timestamp": "t1", "files": [{"name": "a", "size_bytes": 1}]}
        )
        storage.record_checkpoint(
            {"step": 50, "timestamp": "t2", "files": [{"name": "b", "size_bytes": 2}]}
        )
        storage.record_checkpoint(
            {"step": 100, "timestamp": "t3", "files": [{"name": "c", "size_bytes": 3}]}
        )

        checkpoints = storage.read_checkpoints()
        assert [c.step for c in checkpoints] == [50, 100]
        assert checkpoints[1].timestamp == "t3"
        assert checkpoints[1].files[0].name == "c"

    def test_read_checkpoints_adds_local_saves(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """Local saves are listed too, and win over the index for their step."""
        run_dir = tmp_path / "run"
        storage = RunStorage(run_dir=run_dir)
        storage.record_checkpoint(
            {
                "step": 10,
                "timestamp": "t-remote",
                "files": [{"name": "a", "size_bytes": 1}],
            }
        )
        storage.record_checkpoint(
            {
                "step": 20,
                "timestamp": "t-remote",
                "files": [{"name": "a", "size_bytes": 1}],
            }
        )
        with mock.patch("time.strftime", return_value="t-local"):
            _save_local(run_dir, 5, state_dict={"w": 1})
            _save_local(run_dir, 10, state_dict={"w": 1})
        stage_checkpoint(run_dir, 30, state_dict={"w": 1})

        checkpoints = storage.read_checkpoints()
        assert [(c.step, c.timestamp) for c in checkpoints] == [
            (5, "t-local"),
            (10, "t-local"),
            (20, "t-remote"),
        ]

    def test_read_checkpoints_accepts_oldest_entry_shape(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """A download of a very old save keeps its bare-name, timestamp-less meta."""
        client, stored = _mock_s3_client()
        stored[f"{_step_prefix('run-1', 5)}/checkpoint.pt"] = pickle.dumps({"w": 1})
        stored[_index_key("run-1")] = json.dumps(
            [{"step": 5, "files": ["checkpoint.pt"], "size_bytes": 100}]
        ).encode()
        _s3_storage(client).load_checkpoint(5)

        (checkpoint,) = RunStorage(
            run_dir=get_run_dir("proj", "run-1")
        ).read_checkpoints()

        assert checkpoint.step == 5
        assert checkpoint.timestamp == ""
        assert [(f.name, f.size_bytes) for f in checkpoint.files] == [
            ("checkpoint.pt", 100)
        ]


class TestS3CheckpointUpload:
    """How a save is laid out in S3 and committed there."""

    def test_path_checkpoint_layout(self, tmp_path: Path) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        user_file = tmp_path / "model.pt"
        user_file.write_bytes(b"fake model data")

        meta = _write_and_upload(storage, step=100, path=str(user_file))

        assert meta is not None
        assert _remote_meta(stored, "run-1", 100) == meta
        assert meta["files"] == [
            {"name": "checkpoint.pt", "size_bytes": len(b"fake model data")}
        ]
        assert _step_keys(stored, "run-1", 100) == [
            f"{meta['save_id']}/checkpoint.pt",
            "meta.json",
        ]
        assert _remote_file(stored, "run-1", 100, "checkpoint.pt") == b"fake model data"
        assert json.loads(stored[_index_key("run-1")]) == [meta]

    def test_state_dict_model_and_optimizer(self, fake_torch: mock.MagicMock) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)

        _write_and_upload(
            storage,
            step=50,
            state_dict={"w": "data"},
            optimizer_state_dict={"lr": 0.01},
        )

        meta = _remote_meta(stored, "run-1", 50)
        assert [f["name"] for f in meta["files"]] == ["model.pt", "optimizer.pt"]
        assert pickle.loads(_remote_file(stored, "run-1", 50, "model.pt")) == {
            "w": "data"
        }
        assert _remote_file(stored, "run-1", 50, "optimizer.pt")

    def test_state_dict_model_only(self, fake_torch: mock.MagicMock) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)

        _write_and_upload(storage, step=10, state_dict={"w": "data"})

        meta = _remote_meta(stored, "run-1", 10)
        assert [f["name"] for f in meta["files"]] == ["model.pt"]
        assert _step_keys(stored, "run-1", 10) == [
            f"{meta['save_id']}/model.pt",
            "meta.json",
        ]

    def test_index_accumulates_steps(self, tmp_path: Path) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"ckpt")

        _write_and_upload(storage, step=100, path=str(user_file))
        _write_and_upload(storage, step=50, path=str(user_file))

        index = json.loads(stored[_index_key("run-1")])
        assert [e["step"] for e in index] == [50, 100]
        assert [c["step"] for c in storage.list_checkpoints()] == [50, 100]

    def test_list_checkpoints_empty(self) -> None:
        client, _ = _mock_s3_client()
        assert _s3_storage(client).list_checkpoints() == []

    def test_upload_returns_false_on_s3_error(self, tmp_path: Path) -> None:
        """A failed upload reports False and leaves the user's file alone."""
        client, _ = _mock_s3_client()
        client.upload_file.side_effect = BotoCoreError()
        storage = _s3_storage(client)
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"data")

        assert _write_and_upload(storage, step=7, path=str(user_file)) is None
        assert user_file.read_bytes() == b"data"

    def test_failed_reupload_leaves_previous_save_in_s3(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """New files go under their own prefix, so S3's committed save is untouched."""
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        first = _write_and_upload(
            storage, step=5, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1}
        )
        upload_file = client.upload_file.side_effect

        def fail_on_optimizer(Filename, Bucket, Key, **kwargs):
            if Key.endswith("optimizer.pt"):
                raise BotoCoreError()
            upload_file(Filename, Bucket, Key, **kwargs)

        client.upload_file.side_effect = fail_on_optimizer
        second = _write_and_upload(
            storage, step=5, state_dict={"w": 2}, optimizer_state_dict={"lr": 0.2}
        )

        assert second is None
        assert _remote_meta(stored, "run-1", 5) == first
        assert pickle.loads(_remote_file(stored, "run-1", 5, "model.pt")) == {"w": 1}

    def test_resave_removes_superseded_files(self, fake_torch: mock.MagicMock) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        prefix = _step_prefix("run-1", 5)
        stored[f"{prefix}/checkpoint.pt"] = b"from before save IDs"
        stored[f"{prefix}/{META_FILE}"] = json.dumps(
            {"step": 5, "timestamp": "t", "files": ["checkpoint.pt"]}
        ).encode()
        _write_and_upload(
            storage, step=5, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1}
        )

        meta = _write_and_upload(storage, step=5, state_dict={"w": 2})

        assert meta is not None
        assert _step_keys(stored, "run-1", 5) == [
            f"{meta['save_id']}/model.pt",
            "meta.json",
        ]

    def test_checkpoint_meta_prefers_step_meta_over_index(self) -> None:
        """The step's meta.json is the commit record; the index can lag behind it."""
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        stored[_index_key("run-1")] = json.dumps(
            [{"step": 5, "save_id": "old", "timestamp": "t", "files": []}]
        ).encode()
        stored[f"{_step_prefix('run-1', 5)}/{META_FILE}"] = json.dumps(
            {"step": 5, "save_id": "new", "timestamp": "t", "files": []}
        ).encode()

        assert storage.find_checkpoint(5)["save_id"] == "new"

    def test_delete_checkpoint_optimizer_removes_only_optimizer(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        _write_and_upload(
            storage, step=42, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.01}
        )
        save_id = _remote_meta(stored, "run-1", 42)["save_id"]

        storage.delete_checkpoint_optimizer(step=42)

        meta = _remote_meta(stored, "run-1", 42)
        assert _step_keys(stored, "run-1", 42) == [f"{save_id}/model.pt", "meta.json"]
        assert [f["name"] for f in meta["files"]] == ["model.pt"]
        assert meta["save_id"] == save_id
        index = json.loads(stored[_index_key("run-1")])
        assert [f["name"] for f in index[0]["files"]] == ["model.pt"]

    def test_delete_checkpoint_optimizer_noop_when_absent(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, stored = _mock_s3_client()
        storage = _s3_storage(client)
        _write_and_upload(storage, step=7, state_dict={"w": 1})

        before = dict(stored)
        storage.delete_checkpoint_optimizer(step=7)

        assert stored == before
        client.delete_object.assert_not_called()

    def test_delete_checkpoint_optimizer_unknown_step_raises(self) -> None:
        client, _ = _mock_s3_client()
        with pytest.raises(FileNotFoundError):
            _s3_storage(client).delete_checkpoint_optimizer(step=999)


class TestS3CheckpointLoad:
    """Loading S3's save of a checkpoint through the local run dir."""

    def test_load_model_and_optimizer(self, fake_torch: mock.MagicMock) -> None:
        client, _ = _mock_s3_client()
        storage = _s3_storage(client)
        _write_and_upload(
            storage,
            step=10,
            state_dict={"w": [1, 2, 3]},
            optimizer_state_dict={"lr": 0.01},
        )

        assert storage.load_checkpoint(10) == {
            "model_state_dict": {"w": [1, 2, 3]},
            "optimizer_state_dict": {"lr": 0.01},
        }

    def test_load_skipping_optimizer_downloads_only_model(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, _ = _mock_s3_client()
        storage = _s3_storage(client)
        _write_and_upload(
            storage, step=10, state_dict={"w": [1]}, optimizer_state_dict={"lr": 0.1}
        )

        assert storage.load_checkpoint(10, load_optimizer=False) == {
            "model_state_dict": {"w": [1]}
        }
        downloaded = [c.kwargs["Key"] for c in client.download_file.call_args_list]
        assert [key.rsplit("/", 1)[1] for key in downloaded] == ["model.pt"]

    def test_load_adds_missing_file_to_same_save(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, _ = _mock_s3_client()
        storage = _s3_storage(client)
        meta = _write_and_upload(
            storage, step=10, state_dict={"w": [1]}, optimizer_state_dict={"lr": 0.1}
        )
        storage.load_checkpoint(10, load_optimizer=False)
        client.download_file.reset_mock()

        assert storage.load_checkpoint(10)["optimizer_state_dict"] == {"lr": 0.1}
        downloaded = [c.kwargs["Key"] for c in client.download_file.call_args_list]
        assert [key.rsplit("/", 1)[1] for key in downloaded] == ["optimizer.pt"]
        local_dir = local_checkpoint_dir("proj", "run-1", 10)
        assert json.loads((local_dir / META_FILE).read_text()) == meta

    def test_load_legacy_format(self, fake_torch: mock.MagicMock) -> None:
        """A save from before save IDs: files at the step root, index entry only."""
        import pickle

        client, stored = _mock_s3_client()
        stored[f"{_step_prefix('run-1', 5)}/checkpoint.pt"] = pickle.dumps(
            {"model_state_dict": {"w": 42}, "optimizer_state_dict": {"lr": 0.01}}
        )
        stored[_index_key("run-1")] = json.dumps(
            [{"step": 5, "files": ["checkpoint.pt"], "size_bytes": 100}]
        ).encode()

        assert _s3_storage(client).load_checkpoint(5) == {
            "model_state_dict": {"w": 42},
            "optimizer_state_dict": {"lr": 0.01},
        }

    def test_load_reuses_local_copy(self, fake_torch: mock.MagicMock) -> None:
        client, _ = _mock_s3_client()
        storage = _s3_storage(client)
        _write_and_upload(storage, step=20, state_dict={"w": 1})

        storage.load_checkpoint(20)
        client.download_file.reset_mock()
        storage.load_checkpoint(20)

        client.download_file.assert_not_called()

    def test_concurrent_loads_of_a_step_share_one_copy(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """Ranks of a distributed job loading the same step must not collide."""
        client, _ = _mock_s3_client()
        _write_and_upload(
            _s3_storage(client),
            step=10,
            state_dict={"w": 1},
            optimizer_state_dict={"lr": 0.1},
        )
        barrier = threading.Barrier(2, timeout=10)
        download = client.download_file.side_effect

        def download_in_step(**kwargs: Any) -> None:
            download(**kwargs)
            barrier.wait()

        client.download_file.side_effect = download_in_step
        results: list[Any] = [None, None]

        def load(i: int) -> None:
            try:
                results[i] = _s3_storage(client).load_checkpoint(10)
            except BaseException as e:
                results[i] = e

        threads = [threading.Thread(target=load, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = {"model_state_dict": {"w": 1}, "optimizer_state_dict": {"lr": 0.1}}
        assert results == [expected, expected]
        assert (
            read_local_checkpoint(local_checkpoint_dir("proj", "run-1", 10)) == expected
        )
        assert _staging_leftovers(get_run_dir("proj", "run-1")) == []

    def test_load_nonexistent_step_raises(self) -> None:
        client, _ = _mock_s3_client()
        with pytest.raises(FileNotFoundError, match="step 999"):
            _s3_storage(client).load_checkpoint(999)

    def test_interrupted_save_is_not_mixed_into_a_load(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """A save killed mid-upload leaves nothing a later load could pick up."""
        client, _ = _mock_s3_client()
        run = _s3_run(client, "killed")
        run.save_checkpoint(5, state_dict={"w": 1})
        client.upload_file.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            run.save_checkpoint(5, state_dict={"w": 2})
        run.finish()

        run_dir = get_run_dir("proj", "killed")
        assert not checkpoint_dir(run_dir, 5).exists()
        assert _staging_leftovers(run_dir) == []
        with mock.patch("boto3.client", return_value=client):
            loaded = extty.load_checkpoint_from(
                "proj", "killed", 5, s3_config=_S3_CONFIG
            )
        assert loaded == {"model_state_dict": {"w": 1}}

    def test_load_leaves_a_different_local_save_alone(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """S3's save is loaded without replacing a local-only save of the step."""
        client, _ = _mock_s3_client()
        run = _s3_run(client, "diverged")
        run.save_checkpoint(5, state_dict={"w": 1}, optimizer_state_dict={"lr": 0.1})
        client.upload_file.side_effect = BotoCoreError()
        run.save_checkpoint(5, state_dict={"w": 2}, optimizer_state_dict={"lr": 0.2})
        run.finish()
        client.upload_file.side_effect = None
        local_dir = local_checkpoint_dir("proj", "diverged", 5)
        local_meta = json.loads((local_dir / META_FILE).read_text())
        (local_dir / "optimizer.pt").unlink()

        with mock.patch("boto3.client", return_value=client):
            loaded = extty.load_checkpoint_from(
                "proj", "diverged", 5, s3_config=_S3_CONFIG
            )

        assert loaded == {
            "model_state_dict": {"w": 1},
            "optimizer_state_dict": {"lr": 0.1},
        }
        assert json.loads((local_dir / META_FILE).read_text()) == local_meta
        assert read_local_checkpoint(local_dir, load_optimizer=False) == {
            "model_state_dict": {"w": 2}
        }
        assert _staging_leftovers(get_run_dir("proj", "diverged")) == []


class TestRunCheckpointsWithS3:
    """What Run.save_checkpoint keeps locally when S3 is configured."""

    def test_upload_success_removes_local_copy(
        self, extty_home: Path, fake_torch: mock.MagicMock
    ) -> None:
        """The local files go; the local index mirrors S3's and lists the step."""
        client, stored = _mock_s3_client()
        run = _s3_run(client, "s3-run")
        run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        run_dir = extty_home / "runs" / "proj" / "s3-run"
        assert _remote_file(stored, "s3-run", 5, "model.pt")
        assert not checkpoint_dir(run_dir, 5).exists()
        assert _staging_leftovers(run_dir) == []
        assert json.loads((run_dir / "checkpoints.json").read_text()) == json.loads(
            stored[_index_key("s3-run")]
        )
        run_data = extty.get_run("proj", "s3-run", local_only=True)
        assert [c.step for c in run_data.checkpoints] == [5]

    def test_keep_local_loads_without_download(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, _ = _mock_s3_client()
        run = _s3_run(client, "kept-run")
        run.save_checkpoint(5, state_dict={"w": 1}, keep_local=True)

        assert run.load_checkpoint(5) == {"model_state_dict": {"w": 1}}
        client.download_file.assert_not_called()
        run.finish()

    def test_uploaded_resave_drops_older_local_copy(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """A kept older save must not shadow the newer one that went to S3."""
        client, _ = _mock_s3_client()
        run = _s3_run(client, "resaved")
        run.save_checkpoint(5, state_dict={"w": 1}, keep_local=True)
        run.save_checkpoint(5, state_dict={"w": 2})

        assert not local_checkpoint_dir("proj", "resaved", 5).exists()
        assert run.load_checkpoint(5) == {"model_state_dict": {"w": 2}}
        run.finish()

    def test_pulled_checkpoint_reloads_offline(
        self, fake_torch: mock.MagicMock
    ) -> None:
        client, _ = _mock_s3_client()
        run = _s3_run(client, "pulled-run")
        run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        with mock.patch("boto3.client", return_value=client):
            first = extty.load_checkpoint_from(
                "proj", "pulled-run", 5, s3_config=_S3_CONFIG
            )
        with mock.patch("boto3.client", side_effect=AssertionError("S3 contacted")):
            second = extty.load_checkpoint_from(
                "proj", "pulled-run", 5, s3_config=_S3_CONFIG
            )

        assert first == second == {"model_state_dict": {"w": 1}}

    @pytest.mark.parametrize(
        "error",
        [
            BotoCoreError(),
            S3UploadFailedError(
                "Failed to upload ckpt.pt: An error occurred (AccessDenied)"
            ),
        ],
        ids=["botocore", "s3-upload-failed"],
    )
    def test_failed_upload_keeps_local_copy(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        error: Exception,
    ) -> None:
        """The save is committed locally and listed, but not recorded as in S3.

        ``upload_file`` reports bucket-side failures (403, missing bucket) as
        ``S3UploadFailedError``, which is not a botocore exception.
        """
        client, _ = _mock_s3_client()
        client.upload_file.side_effect = error
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"fake model data")

        with caplog.at_level(logging.ERROR, logger="extty"):
            run = _s3_run(client, "run-2")
            run.save_checkpoint(100, path=str(user_file))
            run.finish()

        run_dir = get_run_dir("proj", "run-2")
        local_dir = checkpoint_dir(run_dir, 100)
        assert (local_dir / "checkpoint.pt").read_bytes() == b"fake model data"
        assert not (run_dir / "checkpoints.json").exists()
        assert "local copy kept at" in caplog.text
        run_data = extty.get_run("proj", "run-2", local_only=True)
        assert [c.step for c in run_data.checkpoints] == [100]

    def test_index_write_failure_keeps_local_copy(
        self, fake_torch: mock.MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = _mock_s3_client()
        put_object = client.put_object.side_effect

        def put_object_except_index(**kwargs: Any) -> None:
            if kwargs["Key"].endswith("/checkpoints.json"):
                raise BotoCoreError()
            put_object(**kwargs)

        client.put_object.side_effect = put_object_except_index
        run = _s3_run(client, "no-index")
        with caplog.at_level(logging.ERROR, logger="extty"):
            run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        local = read_local_checkpoint(local_checkpoint_dir("proj", "no-index", 5))
        assert local == {"model_state_dict": {"w": 1}}
        assert "local copy kept at" in caplog.text

    def test_unreadable_remote_index_is_not_overwritten(self, tmp_path: Path) -> None:
        """A failed index read must not rewrite the index with one entry."""
        client, stored = _mock_s3_client()
        index_key = _index_key("run-1")
        stored[index_key] = json.dumps(
            [{"step": 1, "timestamp": "t", "files": []}]
        ).encode()
        get_object = client.get_object.side_effect

        def get_object_denying_index(Bucket: str, Key: str) -> Any:
            if Key == index_key:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "denied"}},
                    "GetObject",
                )
            return get_object(Bucket=Bucket, Key=Key)

        client.get_object.side_effect = get_object_denying_index
        storage = _s3_storage(client)
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"weights")

        assert _write_and_upload(storage, step=2, path=str(user_file)) is None
        assert [e["step"] for e in json.loads(stored[index_key])] == [1]

    def test_path_checkpoint_uploads_without_local_copy(self, tmp_path: Path) -> None:
        """A saved file goes to S3 from where it is; nothing is copied locally."""
        client, stored = _mock_s3_client()
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"weights")
        run = _s3_run(client, "path-run")
        run.save_checkpoint(5, path=str(user_file))
        run.finish()

        uploaded_from = [
            c.kwargs["Filename"] for c in client.upload_file.call_args_list
        ]
        assert uploaded_from == [str(user_file)]
        assert _remote_file(stored, "path-run", 5, "checkpoint.pt") == b"weights"
        assert not local_checkpoint_dir("proj", "path-run", 5).exists()

    def test_path_checkpoint_keep_local_copies_file(
        self, tmp_path: Path, fake_torch: mock.MagicMock
    ) -> None:
        """With keep_local, the saved file is copied in, independent of the original."""
        client, _ = _mock_s3_client()
        user_file = tmp_path / "ckpt.pt"
        _pickle_save({"model_state_dict": {"w": 1}}, user_file)
        run = _s3_run(client, "kept-path")
        run.save_checkpoint(5, path=str(user_file), keep_local=True)
        user_file.unlink()

        assert run.load_checkpoint(5) == {"model_state_dict": {"w": 1}}
        client.download_file.assert_not_called()
        run.finish()


class TestDeleteLocalCheckpoint:
    """delete_local_checkpoint only removes a copy S3 also has."""

    def test_noop_when_absent(self) -> None:
        assert extty.delete_local_checkpoint("proj", "run-empty", step=42) is False

    def test_refuses_only_copy_without_s3(self, fake_torch: mock.MagicMock) -> None:
        run = extty.run.Run("proj", name="only-copy", system_metrics=False)
        run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        with pytest.raises(RuntimeError, match="S3 is not configured"):
            extty.delete_local_checkpoint("proj", "only-copy", 5)
        assert extty.load_checkpoint_from("proj", "only-copy", 5) == {
            "model_state_dict": {"w": 1}
        }

        assert extty.delete_local_checkpoint("proj", "only-copy", 5, force=True)
        with pytest.raises(FileNotFoundError):
            extty.load_checkpoint_from("proj", "only-copy", 5)

    def test_deletes_when_s3_has_same_save(self, fake_torch: mock.MagicMock) -> None:
        client, _ = _mock_s3_client()
        run = _s3_run(client, "backed-up")
        run.save_checkpoint(5, state_dict={"w": 1}, keep_local=True)
        run.finish()

        with mock.patch("boto3.client", return_value=client):
            removed = extty.delete_local_checkpoint(
                "proj", "backed-up", 5, s3_config=_S3_CONFIG
            )

        assert removed is True
        assert not local_checkpoint_dir("proj", "backed-up", 5).exists()

    def test_refuses_when_step_not_in_s3(self, fake_torch: mock.MagicMock) -> None:
        client, _ = _mock_s3_client()
        client.upload_file.side_effect = BotoCoreError()
        run = _s3_run(client, "failed-upload")
        run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        with (
            mock.patch("boto3.client", return_value=client),
            pytest.raises(RuntimeError, match="not in S3"),
        ):
            extty.delete_local_checkpoint(
                "proj", "failed-upload", 5, s3_config=_S3_CONFIG
            )
        assert local_checkpoint_dir("proj", "failed-upload", 5).exists()

    def test_refuses_when_s3_has_different_save(
        self, fake_torch: mock.MagicMock
    ) -> None:
        """S3's save of a step doesn't vouch for a local-only re-save, even one
        made in the same second with the same file sizes."""
        client, _ = _mock_s3_client()
        run = _s3_run(client, "resumed")
        with mock.patch("time.strftime", return_value="2026-09-30T10:00:00+0000"):
            run.save_checkpoint(5, state_dict={"w": 1})
            client.upload_file.side_effect = BotoCoreError()
            run.save_checkpoint(5, state_dict={"w": 2})
        run.finish()

        with (
            mock.patch("boto3.client", return_value=client),
            pytest.raises(RuntimeError, match="different save"),
        ):
            extty.delete_local_checkpoint("proj", "resumed", 5, s3_config=_S3_CONFIG)

        assert extty.load_checkpoint_from("proj", "resumed", 5) == {
            "model_state_dict": {"w": 2}
        }

    def test_refuses_when_s3_unreachable(self, fake_torch: mock.MagicMock) -> None:
        client, _ = _mock_s3_client()
        run = _s3_run(client, "offline")
        run.save_checkpoint(5, state_dict={"w": 1}, keep_local=True)
        run.finish()
        client.get_object.side_effect = EndpointConnectionError(
            endpoint_url="https://s3.example.com"
        )

        with (
            mock.patch("boto3.client", return_value=client),
            pytest.raises(RuntimeError, match="Could not check S3"),
        ):
            extty.delete_local_checkpoint("proj", "offline", 5, s3_config=_S3_CONFIG)

        assert local_checkpoint_dir("proj", "offline", 5).exists()

    def test_allows_download_cache_without_meta(self, tmp_path: Path) -> None:
        """Download caches from before meta.json was written locally stay deletable."""
        client, _ = _mock_s3_client()
        user_file = tmp_path / "ckpt.pt"
        user_file.write_bytes(b"weights")
        run = _s3_run(client, "old-cache")
        run.save_checkpoint(5, path=str(user_file))
        run.finish()
        cache = local_checkpoint_dir("proj", "old-cache", 5)
        cache.mkdir(parents=True)
        (cache / "checkpoint.pt").write_bytes(b"weights")

        with mock.patch("boto3.client", return_value=client):
            removed = extty.delete_local_checkpoint(
                "proj", "old-cache", 5, s3_config=_S3_CONFIG
            )

        assert removed is True
        assert not cache.exists()

    def test_refuses_without_boto3(self, fake_torch: mock.MagicMock) -> None:
        run = extty.run.Run("proj", name="no-boto", system_metrics=False)
        run.save_checkpoint(5, state_dict={"w": 1})
        run.finish()

        with (
            mock.patch(
                "extty.s3._make_s3_client",
                side_effect=ImportError("boto3 is required for S3 storage."),
            ),
            pytest.raises(RuntimeError, match="Could not check S3"),
        ):
            extty.delete_local_checkpoint(
                "proj", "no-boto", 5, s3_config=S3Config(bucket="b")
            )

        assert local_checkpoint_dir("proj", "no-boto", 5).exists()


class TestExttyHome:
    """Tests for relocating extty's local data with ``EXTTY_HOME``."""

    def test_env_var_relocates_local_data(self, extty_home: Path) -> None:
        assert get_extty_home() == extty_home
        assert get_runs_dir() == extty_home / "runs"
        assert get_artifacts_dir() == extty_home / "artifacts"

    def test_defaults_to_dot_extty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXTTY_HOME")
        assert get_extty_home() == Path.home() / ".extty"

    def test_empty_value_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXTTY_HOME", "")
        assert get_extty_home() == Path.home() / ".extty"

    def test_s3_config_file_read_from_extty_home(self, extty_home: Path) -> None:
        (extty_home / "s3").mkdir(parents=True)
        (extty_home / "s3" / "config.toml").write_text('bucket = "home-bucket"\n')

        config = S3Config.load()

        assert config is not None
        assert config.bucket == "home-bucket"


class TestRunDataReading:
    """Tests for RunData, get_run(), and get_runs() read-path API."""

    def _mock_runs_dir(self, tmp_path: Path):
        """Context manager that points extty's runs dir into ``tmp_path``."""
        return _runs_dir_at(tmp_path / "runs")

    def test_get_run_loads_metadata(self, tmp_path: Path) -> None:
        """Test that get_run returns RunData with correct metadata."""
        with self._mock_runs_dir(tmp_path):
            extty.init(
                "myproject", name="run-1", config={"lr": 0.001}, system_metrics=False
            )
            extty.log({"train/loss": 0.5}, step=0)
            extty.finish()

            run = extty.get_run("myproject", "run-1")
            assert run.project == "myproject"
            assert run.name == "run-1"
            assert run.config["lr"] == 0.001
            assert run.status == "completed"
            assert run.finished_at is not None

    def test_get_run_not_found_raises(self, tmp_path: Path) -> None:
        """Test that get_run raises FileNotFoundError for missing runs."""
        with self._mock_runs_dir(tmp_path):
            with pytest.raises(FileNotFoundError):
                extty.get_run("nonexistent", "no-run")

    def test_get_runs_returns_all(self, tmp_path: Path) -> None:
        """Test that get_runs returns all runs across projects."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj-a", name="run-1", system_metrics=False)
            extty.finish()
            extty.init("proj-b", name="run-2", system_metrics=False)
            extty.finish()

            runs = extty.get_runs()
            assert len(runs) == 2
            projects = {r.project for r in runs}
            assert projects == {"proj-a", "proj-b"}

    def test_get_runs_filters_by_project(self, tmp_path: Path) -> None:
        """Test that get_runs filters by project name."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj-a", name="run-1", system_metrics=False)
            extty.finish()
            extty.init("proj-b", name="run-2", system_metrics=False)
            extty.finish()

            runs = extty.get_runs(project="proj-a")
            assert len(runs) == 1
            assert runs[0].project == "proj-a"

    def test_get_runs_sorted_by_started_at(self, tmp_path: Path) -> None:
        """Test that get_runs returns runs sorted most recent first."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="older-run", system_metrics=False)
            extty.finish()
            extty.init("proj", name="newer-run", system_metrics=False)
            extty.finish()

            runs = extty.get_runs(project="proj")
            assert len(runs) == 2
            assert runs[0].name == "newer-run"
            assert runs[1].name == "older-run"

    def test_get_runs_empty_dir(self, tmp_path: Path) -> None:
        """Test get_runs with no runs dir returns empty list."""
        with _runs_dir_at(tmp_path / "nonexistent" / "runs"):
            assert extty.get_runs() == []

    def test_get_runs_skips_corrupt_meta(self, tmp_path: Path) -> None:
        """Test that get_runs skips runs with corrupt meta.json."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="good-run", system_metrics=False)
            extty.finish()

            corrupt_dir = tmp_path / "runs" / "proj" / "bad-run"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / "meta.json").write_text("{invalid json")

            import warnings

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                runs = extty.get_runs(project="proj")
            assert len(runs) == 1
            assert runs[0].name == "good-run"
            assert any("invalid meta.json" in str(warning.message) for warning in w)

    def test_metric_names(self, tmp_path: Path) -> None:
        """Test that metric_names discovers all logged metrics."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.log({"train/loss": 0.5, "train/acc": 0.8}, step=0)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            names = run.metric_names
            assert "train/loss" in names
            assert "train/acc" in names

    def test_metric_returns_points(self, tmp_path: Path) -> None:
        """Test that metric() returns correct MetricPoint values."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.log({"loss": 0.5}, step=0)
            extty.log({"loss": 0.3}, step=1)
            extty.log({"loss": 0.1}, step=2)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            points = run.metric("loss")
            assert len(points) == 3
            assert points[0].step == 0
            assert points[0].value == 0.5
            assert points[2].step == 2
            assert points[2].value == 0.1

    def test_metric_not_found_raises(self, tmp_path: Path) -> None:
        """Test that metric() raises FileNotFoundError for missing metrics."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            with pytest.raises(FileNotFoundError):
                run.metric("nonexistent")

    def test_system_metrics(self, tmp_path: Path) -> None:
        """Test reading system metrics from a run."""
        with self._mock_runs_dir(tmp_path):
            run_dir = tmp_path / "runs" / "proj" / "run-1"
            storage = RunStorage(run_dir=run_dir)
            meta = MetaData(
                project="proj",
                run_name="run-1",
                config={},
                started_at="2024-01-01T00:00:00Z",
                status="completed",
            )
            storage.write_meta(meta)
            storage.log_system(8.0, 32.0, 4.0, 24.0, 50.0)
            storage.log_system(9.0, 32.0, 5.0, 24.0, 60.0)

            run = extty.get_run("proj", "run-1")
            sys_metrics = run.system_metrics
            assert len(sys_metrics) == 2
            assert sys_metrics[0].ram_used_gb == 8.0
            assert sys_metrics[0].gpu_util_pct == 50.0
            assert sys_metrics[1].ram_used_gb == 9.0

    def test_system_metrics_empty(self, tmp_path: Path) -> None:
        """Test that system_metrics returns empty list when no system.csv."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            assert run.system_metrics == []

    def test_example_names_and_data(self, tmp_path: Path) -> None:
        """Test reading example data from a run."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.log(
                {"val/example": extty.Example(prompt="Hello", responses=["Hi"])},
                step=0,
            )
            extty.finish()

            run = extty.get_run("proj", "run-1")
            assert "val/example" in run.example_names
            examples = run.examples("val/example")
            assert len(examples) == 1
            assert examples[0].step == 0
            assert examples[0].data["prompt"] == ["Hello"]

    def test_duration_seconds(self, tmp_path: Path) -> None:
        """Test duration_seconds computation."""
        with self._mock_runs_dir(tmp_path):
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            assert run.duration_seconds is not None
            assert run.duration_seconds >= 0

    def test_duration_seconds_none_when_running(self, tmp_path: Path) -> None:
        """Test duration_seconds returns None for unfinished runs."""
        with self._mock_runs_dir(tmp_path):
            run_dir = tmp_path / "runs" / "proj" / "run-1"
            storage = RunStorage(run_dir=run_dir)
            meta = MetaData(
                project="proj",
                run_name="run-1",
                config={},
                started_at="2024-01-01T00:00:00Z",
                status="running",
            )
            storage.write_meta(meta)

            run = extty.get_run("proj", "run-1")
            assert run.duration_seconds is None


class TestS3FailureTolerance:
    """Tests that S3 write failures are non-fatal and properly handled."""

    def _make_mock_s3_client(
        self, stored: dict[str, bytes] | None = None
    ) -> tuple[mock.MagicMock, dict[str, bytes]]:
        if stored is None:
            stored = {}
        client = mock.MagicMock()

        def put_object(Bucket, Key, Body, ContentType=None):
            if isinstance(Body, str):
                Body = Body.encode("utf-8")
            stored[Key] = Body

        def get_object(Bucket, Key):
            if Key in stored:
                body = mock.MagicMock()
                body.read.return_value = stored[Key]
                return {"Body": body}
            raise client.exceptions.NoSuchKey(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            )

        def upload_file(Filename, Bucket, Key, **kwargs):
            with open(Filename, "rb") as f:
                stored[Key] = f.read()

        client.put_object.side_effect = put_object
        client.get_object.side_effect = get_object
        client.upload_file.side_effect = upload_file
        client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        return client, stored

    def _make_storage(
        self,
        client: mock.MagicMock,
        bucket: str = "test-bucket",
        prefix: str = "test",
        project: str = "myproject",
        run_name: str = "run-001",
    ) -> S3Storage:
        config = S3Config(bucket=bucket, prefix=prefix)
        with mock.patch("boto3.client", return_value=client):
            storage = S3Storage(config, project, run_name)
        return storage

    def test_flush_does_not_raise_on_upload_failure(self, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        client.put_object.side_effect = BotoCoreError()

        storage.log_metric("loss", 0.5, step=1)
        storage._buffer_max_count = 1

        with caplog.at_level(logging.WARNING, logger="extty"):
            storage.flush()

        assert "Failed to upload metrics 'loss' to S3" in caplog.text

    def test_failed_metrics_retained_in_buffer(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_metric("loss", 0.5, step=1)
        storage.log_metric("loss", 0.4, step=2)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert "loss" in storage._metric_buffer
        assert len(storage._metric_buffer["loss"]) == 2
        assert storage._buffer_count == 2
        assert storage._cumulative_metric_rows.get("loss", []) == []

    def test_failed_examples_retained_in_buffer(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_example("outputs", {"text": "hello"}, step=1)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert "outputs" in storage._example_buffer
        assert len(storage._example_buffer["outputs"]) == 1
        assert storage._cumulative_example_rows.get("outputs", []) == []

    def test_failed_system_metrics_retained_in_buffer(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_system(4.0, 16.0)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert len(storage._system_buffer) == 1
        assert storage._cumulative_system_rows == []

    def test_no_data_loss_after_transient_failure(self) -> None:
        """S3 fails on first flush, recovers on second — all data arrives."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_metric("loss", 0.5, step=1)
        storage.log_metric("loss", 0.4, step=2)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert "loss" in storage._metric_buffer

        storage.log_metric("loss", 0.3, step=3)

        original_put = lambda Bucket, Key, Body, ContentType=None: stored.__setitem__(  # noqa: E731
            Key, Body if isinstance(Body, bytes) else Body.encode("utf-8")
        )
        client.put_object.side_effect = original_put
        client.get_object.side_effect = lambda Bucket, Key: (_ for _ in ()).throw(
            client.exceptions.NoSuchKey({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        )
        storage.flush()

        assert "loss" not in storage._metric_buffer
        assert storage._buffer_count == 0

        metrics_key = "test/runs/myproject/run-001/metrics/loss.csv"
        assert metrics_key in stored
        csv_content = stored[metrics_key].decode("utf-8")
        lines = csv_content.strip().split("\n")
        assert len(lines) == 4
        data_lines = lines[1:]
        steps = [line.split(",")[0] for line in data_lines]
        assert steps == ["1", "2", "3"]

    def test_write_meta_does_not_raise(self, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        client.put_object.side_effect = BotoCoreError()

        with caplog.at_level(logging.WARNING, logger="extty"):
            storage.write_meta({"project": "test", "status": "running"})

        assert "Failed to write run metadata to S3" in caplog.text

    def test_save_checkpoint_does_not_raise(self, tmp_path, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        fake_file = tmp_path / "model.pt"
        fake_file.write_bytes(b"fake model data")

        client.upload_file.side_effect = BotoCoreError()

        with caplog.at_level(logging.WARNING, logger="extty"):
            _write_and_upload(storage, step=100, path=str(fake_file))

        assert "Failed to save checkpoint (step 100) to S3" in caplog.text

    def test_update_checkpoints_index_does_not_raise(self, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        client.put_object.side_effect = BotoCoreError()

        with caplog.at_level(logging.WARNING, logger="extty"):
            storage._update_checkpoints_index(
                {"step": 1, "timestamp": "now", "files": []}
            )

        assert "Failed to update checkpoints index in S3" in caplog.text

    def test_programming_errors_still_propagate(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_metric("loss", 0.5, step=1)

        client.put_object.side_effect = TypeError("bad argument")

        with pytest.raises(TypeError, match="bad argument"):
            storage.flush()

    def test_checkpoint_meta_fallback_on_missing_index(self) -> None:
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        meta_entry = {
            "step": 42,
            "timestamp": "2024-01-01",
            "files": [{"name": "model.pt", "size_bytes": 100}],
        }
        meta_key = "test/runs/myproject/run-001/checkpoints/42/meta.json"
        stored[meta_key] = json.dumps(meta_entry).encode("utf-8")

        result = storage._checkpoint_meta(42)
        assert result["step"] == 42
        assert result["files"][0]["name"] == "model.pt"

    def test_checkpoint_meta_fallback_raises_when_both_missing(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        with pytest.raises(FileNotFoundError, match="Checkpoint step 99 not found"):
            storage._checkpoint_meta(99)

    def test_partial_flush_failure_does_not_block_other_uploads(self) -> None:
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_metric("loss", 0.5, step=1)
        storage.log_system(4.0, 16.0)

        call_count = 0
        original_put = client.put_object.side_effect

        def fail_first_put(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BotoCoreError()
            return original_put(**kwargs)

        client.put_object.side_effect = fail_first_put

        storage.flush()

        assert "loss" in storage._metric_buffer
        system_key = "test/runs/myproject/run-001/system.csv"
        assert system_key in stored

    def test_upload_checkpoint_propagates_local_fs_errors(self, tmp_path) -> None:
        """Local filesystem errors (e.g. missing file) should NOT be swallowed."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)
        entry = {
            "step": 100,
            "timestamp": "t",
            "files": [{"name": "model.pt", "size_bytes": 1}],
        }

        with pytest.raises(FileNotFoundError):
            storage.upload_checkpoint(entry, {"model.pt": tmp_path / "missing.pt"})

    def test_checkpoint_meta_fallback_propagates_non_404_errors(self) -> None:
        """AccessDenied or other S3 errors should propagate, not become FileNotFoundError."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        access_denied = type("AccessDenied", (Exception,), {})
        client.get_object.side_effect = access_denied("forbidden")

        with pytest.raises(access_denied):
            storage._checkpoint_meta(42)

    def test_flush_backoff_defers_maybe_flush(self) -> None:
        """After a failure, _maybe_flush skips until the backoff interval elapses."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)
        storage._buffer_max_seconds = 10.0
        storage._buffer_max_count = 1000

        client.put_object.side_effect = BotoCoreError()

        storage.log_metric("loss", 0.5, step=1)
        storage.flush()
        assert storage._consecutive_failures == 1

        storage.log_metric("loss", 0.4, step=2)

        with mock.patch("extty.s3.time") as mock_time:
            mock_time.time.return_value = storage._last_flush + 15.0
            storage._maybe_flush()

        assert storage._consecutive_failures == 1

        with mock.patch("extty.s3.time") as mock_time:
            mock_time.time.return_value = storage._last_flush + 25.0
            storage._maybe_flush()

        assert storage._consecutive_failures == 2

    def test_flush_backoff_resets_on_success(self) -> None:
        """Successful flush resets the backoff counter."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        client.put_object.side_effect = BotoCoreError()
        storage.log_metric("loss", 0.5, step=1)
        storage.flush()
        storage.flush()
        assert storage._consecutive_failures == 2

        original_put = lambda Bucket, Key, Body, ContentType=None: stored.__setitem__(  # noqa: E731
            Key, Body if isinstance(Body, bytes) else Body.encode("utf-8")
        )
        client.put_object.side_effect = original_put
        client.get_object.side_effect = lambda Bucket, Key: (_ for _ in ()).throw(
            client.exceptions.NoSuchKey({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        )
        storage.flush()
        assert storage._consecutive_failures == 0


class TestS3UploadAvoidsReads:
    """S3 uploads must never call get_object — cumulative state lives in memory."""

    def _make_storage(self) -> tuple[mock.MagicMock, dict[str, bytes], S3Storage]:
        stored: dict[str, bytes] = {}
        client = mock.MagicMock()

        def put_object(Bucket, Key, Body, ContentType=None):
            if isinstance(Body, str):
                Body = Body.encode("utf-8")
            stored[Key] = Body

        def get_object(Bucket, Key):
            raise AssertionError(
                f"unexpected get_object call for {Key!r}: uploads must not read"
            )

        client.put_object.side_effect = put_object
        client.get_object.side_effect = get_object
        client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        config = S3Config(bucket="test-bucket", prefix="test")
        with mock.patch("boto3.client", return_value=client):
            storage = S3Storage(config, "myproject", "run-001")
        return client, stored, storage

    def test_repeated_flush_does_not_call_get_object(self) -> None:
        client, stored, storage = self._make_storage()

        for i in range(20):
            storage.log_metric("loss", float(i), step=i)
            storage.log_example("outputs", {"text": f"hi-{i}"}, step=i)
        storage.flush()
        for i in range(20, 40):
            storage.log_metric("loss", float(i), step=i)
        storage.flush()

        client.get_object.assert_not_called()

        metrics_key = "test/runs/myproject/run-001/metrics/loss.csv"
        assert metrics_key in stored
        lines = stored[metrics_key].decode("utf-8").strip().split("\n")
        steps = [int(line.split(",")[0]) for line in lines[1:]]
        assert steps == list(range(40))

    def test_cumulative_state_drives_full_file_uploads(self) -> None:
        _, stored, storage = self._make_storage()

        for i in range(5):
            storage.log_metric("loss", float(i), step=i)
        storage.flush()
        for i in range(5, 8):
            storage.log_metric("loss", float(i), step=i)
        storage.flush()

        key = "test/runs/myproject/run-001/metrics/loss.csv"
        lines = stored[key].decode("utf-8").strip().split("\n")
        assert len(lines) == 1 + 8
        assert storage._cumulative_metric_rows["loss"]
        assert len(storage._cumulative_metric_rows["loss"]) == 8

    def test_image_upload_writes_png_and_cumulative_index(self) -> None:
        from PIL import Image as PILImage

        client, stored, storage = self._make_storage()

        for step in (0, 1):
            storage.log_image(
                "val/dets", extty.Image(PILImage.new("RGB", (2, 2), "red")), step=step
            )
        storage.flush()
        storage.log_image(
            "val/dets", extty.Image(PILImage.new("RGB", (2, 2), "blue")), step=2
        )
        storage.flush()

        client.get_object.assert_not_called()

        prefix = "test/runs/myproject/run-001/images"
        for step in range(3):
            assert stored[f"{prefix}/val/dets/step_{step}.png"].startswith(b"\x89PNG")

        index_lines = stored[f"{prefix}/val/dets.jsonl"].decode().strip().split("\n")
        assert [json.loads(line)["step"] for line in index_lines] == [0, 1, 2]

    """``extty.get_run`` reads runs directly from S3 when not local."""

    def _seed_s3_client(self, stored: dict[str, bytes]) -> mock.MagicMock:
        client = mock.MagicMock()
        client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})

        def get_object(Bucket, Key):
            if Key in stored:
                body = mock.MagicMock()
                body.read.return_value = stored[Key]
                return {"Body": body}
            raise client.exceptions.NoSuchKey(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            )

        def paginate(Bucket, Prefix=""):
            contents = [
                {"Key": k, "Size": len(v)}
                for k, v in stored.items()
                if k.startswith(Prefix)
            ]
            return iter([{"Contents": contents}])

        paginator = mock.MagicMock()
        paginator.paginate.side_effect = paginate
        client.get_object.side_effect = get_object
        client.get_paginator.return_value = paginator
        return client

    def _seed_remote_run(
        self,
        bucket: str = "test-bucket",
        prefix: str = "test",
        project: str = "myproject",
        run_name: str = "run-remote",
    ) -> tuple[dict[str, bytes], S3Config]:
        run_prefix = f"{prefix}/runs/{project}/{run_name}"
        meta = {
            "project": project,
            "run_name": run_name,
            "config": {"lr": 0.001},
            "started_at": "2024-01-01T00:00:00",
            "finished_at": "2024-01-01T00:01:00",
            "status": "completed",
        }
        metric_csv = "step,timestamp,value\n0,1700000000.0,0.5\n1,1700000001.0,0.3\n"
        examples_jsonl = (
            json.dumps({"step": 0, "timestamp": 1700000000.0, "data": {"prompt": "hi"}})
            + "\n"
        )
        stored: dict[str, bytes] = {
            f"{run_prefix}/meta.json": json.dumps(meta).encode("utf-8"),
            f"{run_prefix}/metrics/loss.csv": metric_csv.encode("utf-8"),
            f"{run_prefix}/examples/val_example.jsonl": examples_jsonl.encode("utf-8"),
        }
        return stored, S3Config(bucket=bucket, prefix=prefix)

    def test_local_run_does_not_touch_s3(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        with (
            _runs_dir_at(runs_dir),
            _runs_dir_at(runs_dir),
        ):
            extty.init(
                "myproject", name="local-run", config={"lr": 0.5}, system_metrics=False
            )
            extty.log({"loss": 0.1}, step=0)
            extty.finish()

            with (
                mock.patch("boto3.client") as boto_mock,
                mock.patch("extty.s3.S3Config.load") as load_mock,
            ):
                run = extty.get_run("myproject", "local-run")
                boto_mock.assert_not_called()
                load_mock.assert_not_called()
            assert run.config["lr"] == 0.5
            assert run.metric("loss")[0].value == 0.1

    def test_s3_fallback_loads_meta_metrics_examples(self, tmp_path: Path) -> None:
        stored, s3_config = self._seed_remote_run(
            project="myproject", run_name="run-remote"
        )
        client = self._seed_s3_client(stored)

        runs_dir = tmp_path / "runs"
        with (
            _runs_dir_at(runs_dir),
            mock.patch("extty.s3.S3Config.load", return_value=s3_config),
            mock.patch("boto3.client", return_value=client),
        ):
            run = extty.get_run("myproject", "run-remote")

        assert run.project == "myproject"
        assert run.name == "run-remote"
        assert run.config == {"lr": 0.001}
        assert run.status == "completed"

        assert sorted(run.metric_names) == ["loss"]
        points = run.metric("loss")
        assert [p.step for p in points] == [0, 1]
        assert [p.value for p in points] == [0.5, 0.3]

        assert run.example_names == ["val_example"]
        examples = run.examples("val_example")
        assert len(examples) == 1
        assert examples[0].data == {"prompt": "hi"}

    def test_local_only_does_not_consult_s3(self, tmp_path: Path) -> None:
        stored, s3_config = self._seed_remote_run(run_name="run-remote")
        client = self._seed_s3_client(stored)

        runs_dir = tmp_path / "runs"
        with (
            _runs_dir_at(runs_dir),
            mock.patch("extty.s3.S3Config.load", return_value=s3_config) as load_mock,
            mock.patch("boto3.client", return_value=client) as boto_mock,
        ):
            with pytest.raises(FileNotFoundError):
                extty.get_run("myproject", "run-remote", local_only=True)
            load_mock.assert_not_called()
            boto_mock.assert_not_called()

    def test_missing_everywhere_raises(self, tmp_path: Path) -> None:
        stored, s3_config = self._seed_remote_run(run_name="run-remote")
        client = self._seed_s3_client(stored)

        runs_dir = tmp_path / "runs"
        with (
            _runs_dir_at(runs_dir),
            mock.patch("extty.s3.S3Config.load", return_value=s3_config),
            mock.patch("boto3.client", return_value=client),
        ):
            with pytest.raises(FileNotFoundError, match="not found locally or in S3"):
                extty.get_run("myproject", "no-such-run")

    def test_no_s3_config_raises_friendly_error(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        with (
            _runs_dir_at(runs_dir),
            mock.patch("extty.s3.S3Config.load", return_value=None),
        ):
            with pytest.raises(FileNotFoundError, match="no S3 configuration"):
                extty.get_run("myproject", "missing-run")


class TestDistributedInit:
    def test_rank_env_nonzero_returns_noop(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with _runs_dir_at(tmp_path / "runs"):
                run = extty.init("test-project", name="dist-test", system_metrics=False)
                assert isinstance(run, extty.NoOpRun)
                extty.finish()
        assert not (tmp_path / "runs").exists()

    def test_rank_env_zero_returns_real_run(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            with _runs_dir_at(tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="rank0-test", system_metrics=False
                )
                assert not isinstance(run, extty.NoOpRun)
                assert isinstance(run, extty.Run)
                extty.finish()

    def test_no_rank_env_returns_real_run(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RANK", None)
            with _runs_dir_at(tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="norank-test", system_metrics=False
                )
                assert not isinstance(run, extty.NoOpRun)
                extty.finish()

    def test_explicit_rank_overrides_env(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            with _runs_dir_at(tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="override-test", system_metrics=False, rank=3
                )
                assert isinstance(run, extty.NoOpRun)
                extty.finish()

        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with _runs_dir_at(tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="override-test2", system_metrics=False, rank=0
                )
                assert not isinstance(run, extty.NoOpRun)
                extty.finish()

    def test_noop_run_log_does_not_crash(self) -> None:
        from extty.run import NoOpRun

        run = NoOpRun("proj", name="test")
        run.log({"loss": 0.5, "acc": 0.9}, step=0)

    def test_noop_run_finish_does_not_crash(self) -> None:
        from extty.run import NoOpRun

        run = NoOpRun("proj", name="test")
        run.finish()

    def test_noop_run_save_checkpoint_does_not_crash(self) -> None:
        from extty.run import NoOpRun

        run = NoOpRun("proj", name="test")
        run.save_checkpoint(step=0)

    def test_noop_run_load_checkpoint_raises(self) -> None:
        from extty.run import NoOpRun

        run = NoOpRun("proj", name="test")
        with pytest.raises(RuntimeError, match="no-op run"):
            run.load_checkpoint(step=0)

    def test_noop_run_context_manager(self) -> None:
        from extty.run import NoOpRun

        with NoOpRun("proj", name="ctx-test") as run:
            run.log({"x": 1}, step=0)

    def test_noop_run_properties(self) -> None:
        from extty.run import NoOpRun

        run = NoOpRun("proj", name="my-name", config={"lr": 0.01})
        assert run.project == "proj"
        assert run.name == "my-name"
        assert run.config == {"lr": 0.01}
        assert run.run_dir == ""

    def test_noop_run_no_directories_created(self, tmp_path: Path) -> None:
        from extty.run import NoOpRun

        with _runs_dir_at(tmp_path / "runs"):
            run = NoOpRun("proj", name="no-dir-test")
            run.log({"x": 1}, step=0)
            run.finish()
        assert not (tmp_path / "runs").exists()

    def test_module_log_and_finish_with_noop(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "2"}):
            with _runs_dir_at(tmp_path / "runs"):
                extty.init("test-project", name="module-noop", system_metrics=False)
                extty.log({"loss": 0.5}, step=0)
                extty.finish()
        assert not (tmp_path / "runs").exists()

    def test_has_active_run_true_for_noop(self) -> None:
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            extty.init("test-project", name="active-check", system_metrics=False)
            assert extty.has_active_run()
            extty.finish()
