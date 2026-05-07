"""Tests for extty library."""

import dataclasses
import json
import logging
import os
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
    generate_random_name,
    sanitize_metric_name,
)
from botocore.exceptions import BotoCoreError
from extty.s3 import S3Config, S3Storage


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
            assert (
                tmp_path / "runs" / "test-project" / "my-run" / "meta.json"
            ).exists()
            extty.finish()

    def test_log_writes_metrics(self, tmp_path: Path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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

    def test_context_manager(self, tmp_path: Path) -> None:
        import json

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            env = {"EXTTY_INSTANCE_ID": "i-abc123", "EXTTY_INSTANCE_PROVIDER": "lambda"}
            with mock.patch.dict(os.environ, env):
                extty.init("test-project", name="inst-test", system_metrics=False)
                extty.finish()

            meta_path = tmp_path / "runs" / "test-project" / "inst-test" / "meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["config"]["_instance_id"] == "lambda:i-abc123"

    def test_instance_id_missing_when_no_env(self, tmp_path: Path) -> None:
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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


class TestLogEvaluation:
    """Tests for log_evaluation functionality."""

    eval_config = {"dataset": "gsm8k", "num_samples": 100}
    model_config = {"n_layers": 2}

    def test_log_evaluation_creates_json(self, tmp_path: Path) -> None:
        """Test that log_evaluation creates a JSON file."""
        with mock.patch(
            "extty.storage.get_models_dir", return_value=tmp_path / "models"
        ):
            extty.log_evaluation(
                "test-project",
                "test-model",
                name="gsm8k",
                metrics={"reward_mean": 0.85, "reward_std": 0.12, "accuracy": 0.78},
                examples=[
                    extty.Example(prompt="What is 2+2?", responses=["4"]),
                    extty.Example(prompt="What is 3*5?", responses=["15"]),
                ],
                eval_config=self.eval_config,
                model_config=self.model_config,
            )

            eval_path = (
                tmp_path
                / "models"
                / "test-project"
                / "test-model"
                / "evaluations"
                / "gsm8k.json"
            )
            assert eval_path.exists()
            content = json.loads(eval_path.read_text())
            assert content["metrics"]["reward_mean"] == 0.85
            assert content["metrics"]["accuracy"] == 0.78
            assert len(content["examples"]) == 2
            assert content["examples"][0]["prompt"] == ["What is 2+2?"]
            assert content["examples"][0]["response"] == [["4"]]
            assert content["config"]["dataset"] == "gsm8k"

    def test_log_evaluation_metrics_only(self, tmp_path: Path) -> None:
        """Test log_evaluation with only metrics."""
        with mock.patch(
            "extty.storage.get_models_dir", return_value=tmp_path / "models"
        ):
            extty.log_evaluation(
                "test-project",
                "model_name",
                name="humaneval",
                metrics={"pass@1": 0.65, "pass@10": 0.82},
            )

            eval_path = (
                tmp_path
                / "models"
                / "test-project"
                / "model_name"
                / "evaluations"
                / "humaneval.json"
            )
            assert eval_path.exists()
            content = json.loads(eval_path.read_text())
            assert content["metrics"]["pass@1"] == 0.65
            assert "examples" not in content
            assert "config" not in content


class TestExperimentDecorator:
    def test_decorator_initializes_and_finishes_run(self, tmp_path: Path) -> None:
        """Test that the decorator properly initializes and finishes a run."""
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

            @extty.experiment("test-project", name="args-test", system_metrics=False)
            def my_experiment(lr: float, epochs: int) -> None:
                pass

            with pytest.warns(UserWarning, match="non-keyword args"):
                my_experiment(0.01, epochs=10)

    def test_decorator_finishes_run_on_exception(self, tmp_path: Path) -> None:
        """Test that the run is finished even when an exception is raised."""
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

            @extty.experiment("test-project", name="return-test", system_metrics=False)
            def my_experiment() -> dict:
                return {"accuracy": 0.95, "loss": 0.05}

            result = my_experiment()

            assert result == {"accuracy": 0.95, "loss": 0.05}

    def test_decorator_name_kwarg(self, tmp_path: Path) -> None:
        """Test that name_kwarg uses a kwarg value as the run name."""
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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
        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):

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


class TestEvaluationDecorator:
    """Tests for the evaluation decorator."""

    def test_decorator_logs_started_and_finished_at(self, tmp_path: Path) -> None:
        """Test that the evaluation decorator logs started_at and finished_at."""
        with mock.patch(
            "extty.storage.get_models_dir", return_value=tmp_path / "models"
        ):

            @extty.evaluation(
                "test-project",
                name="timing-test",
                model="test-model",
                model_config_kwargs=[],
                eval_config_kwargs=[],
            )
            def my_evaluation() -> tuple[dict[str, float], list[extty.Example]]:
                return {"accuracy": 0.95}, []

            my_evaluation()

            eval_path = (
                tmp_path
                / "models"
                / "test-project"
                / "test-model"
                / "evaluations"
                / "timing-test.json"
            )
            assert eval_path.exists()
            content = json.loads(eval_path.read_text())
            assert "started_at" in content
            assert "finished_at" in content
            assert "logged_at" in content
            assert content["started_at"] <= content["finished_at"]
            assert content["finished_at"] <= content["logged_at"]


class TestSaveCheckpoint:
    """Tests for save_checkpoint functionality."""

    def _make_mock_s3_client(
        self, stored: dict[str, bytes] | None = None
    ) -> mock.MagicMock:
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

        def upload_file(Filename, Bucket, Key, ExtraArgs=None):
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

    def test_save_checkpoint_with_path(self, tmp_path: Path) -> None:
        """Test save_checkpoint uploads file and updates checkpoints.json."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(
            client, prefix="test", project="myproject", run_name="run-001"
        )

        fake_file = tmp_path / "model.pt"
        fake_file.write_bytes(b"fake model data")

        storage.save_checkpoint(step=100, path=str(fake_file))

        checkpoint_key = "test/runs/myproject/run-001/checkpoints/100/checkpoint.pt"
        assert checkpoint_key in stored
        assert stored[checkpoint_key] == b"fake model data"

        meta_key = "test/runs/myproject/run-001/checkpoints/100/meta.json"
        assert meta_key in stored
        meta = json.loads(stored[meta_key])
        assert meta["step"] == 100
        assert meta["files"] == [
            {"name": "checkpoint.pt", "size_bytes": len(b"fake model data")}
        ]

        index_key = "test/runs/myproject/run-001/checkpoints.json"
        assert index_key in stored
        index = json.loads(stored[index_key])
        assert len(index) == 1
        assert index[0]["step"] == 100

    def test_save_checkpoint_state_dict_model_and_optimizer(
        self, tmp_path: Path
    ) -> None:
        """Test state_dict mode produces model.pt and optimizer.pt."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(
            client, prefix="pfx", project="proj", run_name="run-1"
        )

        mock_torch = mock.MagicMock()

        def fake_save(obj, path):
            import pickle

            with open(path, "wb") as f:
                pickle.dump(obj, f)

        mock_torch.save.side_effect = fake_save

        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            storage.save_checkpoint(
                step=50,
                state_dict={"weight": "data"},
                optimizer_state_dict={"lr": 0.01},
            )

        model_key = "pfx/runs/proj/run-1/checkpoints/50/model.pt"
        opt_key = "pfx/runs/proj/run-1/checkpoints/50/optimizer.pt"
        assert model_key in stored
        assert opt_key in stored

        meta_key = "pfx/runs/proj/run-1/checkpoints/50/meta.json"
        meta = json.loads(stored[meta_key])
        assert meta["step"] == 50
        file_names = [f["name"] for f in meta["files"]]
        assert file_names == ["model.pt", "optimizer.pt"]
        for f in meta["files"]:
            assert "size_bytes" in f
            assert f["size_bytes"] > 0

    def test_save_checkpoint_state_dict_model_only(self, tmp_path: Path) -> None:
        """Test state_dict mode without optimizer produces only model.pt."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(
            client, prefix="pfx", project="proj", run_name="run-2"
        )

        mock_torch = mock.MagicMock()

        def fake_save(obj, path):
            import pickle

            with open(path, "wb") as f:
                pickle.dump(obj, f)

        mock_torch.save.side_effect = fake_save

        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            storage.save_checkpoint(step=10, state_dict={"weight": "data"})

        model_key = "pfx/runs/proj/run-2/checkpoints/10/model.pt"
        opt_key = "pfx/runs/proj/run-2/checkpoints/10/optimizer.pt"
        assert model_key in stored
        assert opt_key not in stored

        meta_key = "pfx/runs/proj/run-2/checkpoints/10/meta.json"
        meta = json.loads(stored[meta_key])
        file_names = [f["name"] for f in meta["files"]]
        assert file_names == ["model.pt"]

    def test_save_checkpoint_updates_existing_index(self, tmp_path: Path) -> None:
        """Test that saving a second checkpoint appends to index."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(
            client, prefix="pfx", project="proj", run_name="run-x"
        )

        file1 = tmp_path / "ckpt1.pt"
        file1.write_bytes(b"ckpt1")
        file2 = tmp_path / "ckpt2.pt"
        file2.write_bytes(b"ckpt2data")

        storage.save_checkpoint(step=50, path=str(file1))
        storage.save_checkpoint(step=100, path=str(file2))

        index_key = "pfx/runs/proj/run-x/checkpoints.json"
        index = json.loads(stored[index_key])
        assert len(index) == 2
        assert index[0]["step"] == 50
        assert index[1]["step"] == 100

    def test_save_checkpoint_requires_exactly_one_source(self) -> None:
        """Test that providing both or neither path and state_dict raises ValueError."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(
            client, bucket="b", prefix="", project="p", run_name="r"
        )

        with pytest.raises(ValueError, match="Exactly one"):
            storage.save_checkpoint(step=1)

        with pytest.raises(ValueError, match="Exactly one"):
            storage.save_checkpoint(step=1, path="/a", state_dict={"k": "v"})

    def test_list_checkpoints_empty(self) -> None:
        """Test list_checkpoints returns empty list when no checkpoints exist."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(
            client, bucket="b", prefix="", project="p", run_name="r"
        )
        assert storage.list_checkpoints() == []

    def test_list_checkpoints_returns_entries(self, tmp_path: Path) -> None:
        """Test list_checkpoints returns saved checkpoint metadata."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client, prefix="pfx", project="p", run_name="r")

        fake_file = tmp_path / "ckpt.pt"
        fake_file.write_bytes(b"data")

        storage.save_checkpoint(step=10, path=str(fake_file))
        result = storage.list_checkpoints()

        assert len(result) == 1
        assert result[0]["step"] == 10

    def test_module_level_save_checkpoint_without_init_raises(self) -> None:
        """Test that calling extty.save_checkpoint without init raises RuntimeError."""
        extty._active_run = None
        with pytest.raises(RuntimeError, match="No active run"):
            extty.save_checkpoint(step=1, path="/nonexistent")

    def test_run_save_checkpoint_without_s3_raises(self, tmp_path: Path) -> None:
        """Test that save_checkpoint raises when no S3 is configured."""
        with (
            mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"),
            mock.patch("extty.s3.S3Config.load", return_value=None),
        ):
            run = extty.init("test-project", name="no-s3-run", system_metrics=False)
            with pytest.raises(RuntimeError, match="S3 storage is not configured"):
                run.save_checkpoint(step=1, path="/nonexistent")
            extty.finish()


class TestLoadCheckpoint:
    """Tests for load_checkpoint functionality."""

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

        def upload_file(Filename, Bucket, Key, ExtraArgs=None):
            with open(Filename, "rb") as f:
                stored[Key] = f.read()

        def download_file(Bucket, Key, Filename, **kwargs):
            if Key not in stored:
                raise client.exceptions.NoSuchKey(
                    {"Error": {"Code": "NoSuchKey"}}, "GetObject"
                )
            data = stored[Key]
            with open(Filename, "wb") as f:
                f.write(data)
            callback = kwargs.get("Callback")
            if callback is not None:
                callback(len(data))

        def head_object(Bucket, Key):
            if Key not in stored:
                raise client.exceptions.NoSuchKey(
                    {"Error": {"Code": "NoSuchKey"}}, "HeadObject"
                )
            return {"ContentLength": len(stored[Key])}

        client.put_object.side_effect = put_object
        client.get_object.side_effect = get_object
        client.upload_file.side_effect = upload_file
        client.download_file.side_effect = download_file
        client.head_object.side_effect = head_object
        client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        return client, stored

    def _make_storage(
        self,
        client: mock.MagicMock,
        bucket: str = "test-bucket",
        prefix: str = "pfx",
        project: str = "proj",
        run_name: str = "run-1",
    ) -> S3Storage:
        config = S3Config(bucket=bucket, prefix=prefix)
        with mock.patch("boto3.client", return_value=client):
            storage = S3Storage(config, project, run_name)
        return storage

    def test_load_new_format_model_and_optimizer(self, tmp_path: Path) -> None:
        """Test loading a new-format checkpoint with model.pt and optimizer.pt."""

        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        mock_torch = mock.MagicMock()
        mock_torch.save.side_effect = lambda obj, path: _pickle_save(obj, path)
        mock_torch.load.side_effect = lambda path, **kw: _pickle_load(path)

        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            storage.save_checkpoint(
                step=10,
                state_dict={"w": [1, 2, 3]},
                optimizer_state_dict={"lr": 0.01},
            )

        with (
            mock.patch.dict("sys.modules", {"torch": mock_torch}),
            mock.patch("extty.storage.get_runs_dir", return_value=tmp_path / "runs"),
        ):
            result = storage.load_checkpoint(10)

        assert "model_state_dict" in result
        assert result["model_state_dict"] == {"w": [1, 2, 3]}
        assert "optimizer_state_dict" in result
        assert result["optimizer_state_dict"] == {"lr": 0.01}

    def test_load_new_format_skip_optimizer(self, tmp_path: Path) -> None:
        """Test loading only model weights, skipping optimizer."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        mock_torch = mock.MagicMock()
        mock_torch.save.side_effect = lambda obj, path: _pickle_save(obj, path)
        mock_torch.load.side_effect = lambda path, **kw: _pickle_load(path)

        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            storage.save_checkpoint(
                step=10,
                state_dict={"w": [1]},
                optimizer_state_dict={"lr": 0.1},
            )

        with (
            mock.patch.dict("sys.modules", {"torch": mock_torch}),
            mock.patch("extty.storage.get_runs_dir", return_value=tmp_path / "runs"),
        ):
            result = storage.load_checkpoint(10, load_optimizer=False)

        assert "model_state_dict" in result
        assert "optimizer_state_dict" not in result

    def test_load_legacy_format(self, tmp_path: Path) -> None:
        """Test loading an old-format checkpoint (single checkpoint.pt)."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        mock_torch = mock.MagicMock()
        mock_torch.save.side_effect = lambda obj, path: _pickle_save(obj, path)
        mock_torch.load.side_effect = lambda path, **kw: _pickle_load(path)

        legacy_data = {
            "model_state_dict": {"w": 42},
            "optimizer_state_dict": {"lr": 0.01},
        }
        import pickle

        s3_key = "pfx/runs/proj/run-1/checkpoints/5/checkpoint.pt"
        stored[s3_key] = pickle.dumps(legacy_data)

        index_key = "pfx/runs/proj/run-1/checkpoints.json"
        stored[index_key] = json.dumps(
            [{"step": 5, "files": ["checkpoint.pt"], "size_bytes": 100}]
        ).encode()

        with (
            mock.patch.dict("sys.modules", {"torch": mock_torch}),
            mock.patch("extty.storage.get_runs_dir", return_value=tmp_path / "runs"),
        ):
            result = storage.load_checkpoint(5)

        assert result["model_state_dict"] == {"w": 42}
        assert result["optimizer_state_dict"] == {"lr": 0.01}

    def test_load_uses_local_cache(self, tmp_path: Path) -> None:
        """Test that a cached file is not re-downloaded."""
        client, stored = self._make_mock_s3_client()
        storage = self._make_storage(client)

        mock_torch = mock.MagicMock()
        mock_torch.save.side_effect = lambda obj, path: _pickle_save(obj, path)
        mock_torch.load.side_effect = lambda path, **kw: _pickle_load(path)

        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            storage.save_checkpoint(step=20, state_dict={"w": 1})

        with (
            mock.patch.dict("sys.modules", {"torch": mock_torch}),
            mock.patch("extty.storage.get_runs_dir", return_value=tmp_path / "runs"),
        ):
            storage.load_checkpoint(20)
            client.download_file.reset_mock()
            storage.load_checkpoint(20)

        client.download_file.assert_not_called()

    def test_load_nonexistent_step_raises(self) -> None:
        """Test that loading a step that doesn't exist raises FileNotFoundError."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        with pytest.raises(FileNotFoundError, match="step 999"):
            storage.load_checkpoint(999)


def _pickle_save(obj, path):
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


class TestRunDataReading:
    """Tests for RunData, get_run(), and get_runs() read-path API."""

    def _mock_runs_dir(self, tmp_path: Path):
        """Context manager that patches get_runs_dir in both modules."""
        runs_dir = tmp_path / "runs"
        return (
            mock.patch("extty.run.get_runs_dir", return_value=runs_dir),
            mock.patch("extty.query.get_runs_dir", return_value=runs_dir),
        )

    def test_get_run_loads_metadata(self, tmp_path: Path) -> None:
        """Test that get_run returns RunData with correct metadata."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        _, m2 = self._mock_runs_dir(tmp_path)
        with m2:
            with pytest.raises(FileNotFoundError):
                extty.get_run("nonexistent", "no-run")

    def test_get_runs_returns_all(self, tmp_path: Path) -> None:
        """Test that get_runs returns all runs across projects."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
            extty.init("proj-a", name="run-1", system_metrics=False)
            extty.finish()
            extty.init("proj-b", name="run-2", system_metrics=False)
            extty.finish()

            runs = extty.get_runs(project="proj-a")
            assert len(runs) == 1
            assert runs[0].project == "proj-a"

    def test_get_runs_sorted_by_started_at(self, tmp_path: Path) -> None:
        """Test that get_runs returns runs sorted most recent first."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        with mock.patch(
            "extty.query.get_runs_dir", return_value=tmp_path / "nonexistent"
        ):
            assert extty.get_runs() == []

    def test_get_runs_skips_corrupt_meta(self, tmp_path: Path) -> None:
        """Test that get_runs skips runs with corrupt meta.json."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
            extty.init("proj", name="run-1", system_metrics=False)
            extty.log({"train/loss": 0.5, "train/acc": 0.8}, step=0)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            names = run.metric_names
            assert "train/loss" in names
            assert "train/acc" in names

    def test_metric_returns_points(self, tmp_path: Path) -> None:
        """Test that metric() returns correct MetricPoint values."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            with pytest.raises(FileNotFoundError):
                run.metric("nonexistent")

    def test_system_metrics(self, tmp_path: Path) -> None:
        """Test reading system metrics from a run."""
        _, m2 = self._mock_runs_dir(tmp_path)
        with m2:
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
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            assert run.system_metrics == []

    def test_example_names_and_data(self, tmp_path: Path) -> None:
        """Test reading example data from a run."""
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
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
        m1, m2 = self._mock_runs_dir(tmp_path)
        with m1, m2:
            extty.init("proj", name="run-1", system_metrics=False)
            extty.finish()

            run = extty.get_run("proj", "run-1")
            assert run.duration_seconds is not None
            assert run.duration_seconds >= 0

    def test_duration_seconds_none_when_running(self, tmp_path: Path) -> None:
        """Test duration_seconds returns None for unfinished runs."""
        _, m2 = self._mock_runs_dir(tmp_path)
        with m2:
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

        def upload_file(Filename, Bucket, Key, ExtraArgs=None):
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

        with caplog.at_level(logging.WARNING, logger="extty.s3"):
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

    def test_failed_examples_retained_in_buffer(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_example("outputs", {"text": "hello"}, step=1)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert "outputs" in storage._example_buffer
        assert len(storage._example_buffer["outputs"]) == 1

    def test_failed_system_metrics_retained_in_buffer(self) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        storage.log_system(4.0, 16.0)

        client.put_object.side_effect = BotoCoreError()
        storage.flush()

        assert len(storage._system_buffer) == 1

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

        with caplog.at_level(logging.WARNING, logger="extty.s3"):
            storage.write_meta({"project": "test", "status": "running"})

        assert "Failed to write run metadata to S3" in caplog.text

    def test_save_checkpoint_does_not_raise(self, tmp_path, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        fake_file = tmp_path / "model.pt"
        fake_file.write_bytes(b"fake model data")

        client.upload_file.side_effect = BotoCoreError()

        with caplog.at_level(logging.WARNING, logger="extty.s3"):
            storage.save_checkpoint(step=100, path=str(fake_file))

        assert "Failed to save checkpoint (step 100) to S3" in caplog.text

    def test_update_checkpoints_index_does_not_raise(self, caplog) -> None:
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        client.put_object.side_effect = BotoCoreError()

        with caplog.at_level(logging.WARNING, logger="extty.s3"):
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

    def test_save_checkpoint_propagates_local_fs_errors(self, tmp_path) -> None:
        """Local filesystem errors (e.g. missing file) should NOT be swallowed."""
        client, _ = self._make_mock_s3_client()
        storage = self._make_storage(client)

        with pytest.raises(FileNotFoundError):
            storage.save_checkpoint(step=100, path="/nonexistent/model.pt")

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


class TestDistributedInit:
    def test_rank_env_nonzero_returns_noop(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
                run = extty.init("test-project", name="dist-test", system_metrics=False)
                assert isinstance(run, extty.NoOpRun)
                extty.finish()
        assert not (tmp_path / "runs").exists()

    def test_rank_env_zero_returns_real_run(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="rank0-test", system_metrics=False
                )
                assert not isinstance(run, extty.NoOpRun)
                assert isinstance(run, extty.Run)
                extty.finish()

    def test_no_rank_env_returns_real_run(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RANK", None)
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="norank-test", system_metrics=False
                )
                assert not isinstance(run, extty.NoOpRun)
                extty.finish()

    def test_explicit_rank_overrides_env(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
                run = extty.init(
                    "test-project", name="override-test", system_metrics=False, rank=3
                )
                assert isinstance(run, extty.NoOpRun)
                extty.finish()

        with mock.patch.dict(os.environ, {"RANK": "1"}):
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
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

        with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
            run = NoOpRun("proj", name="no-dir-test")
            run.log({"x": 1}, step=0)
            run.finish()
        assert not (tmp_path / "runs").exists()

    def test_module_log_and_finish_with_noop(self, tmp_path: Path) -> None:
        with mock.patch.dict(os.environ, {"RANK": "2"}):
            with mock.patch("extty.run.get_runs_dir", return_value=tmp_path / "runs"):
                extty.init("test-project", name="module-noop", system_metrics=False)
                extty.log({"loss": 0.5}, step=0)
                extty.finish()
        assert not (tmp_path / "runs").exists()

    def test_has_active_run_true_for_noop(self) -> None:
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            extty.init("test-project", name="active-check", system_metrics=False)
            assert extty.has_active_run()
            extty.finish()
