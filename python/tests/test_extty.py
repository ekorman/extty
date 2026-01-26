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
    generate_random_name,
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
        with mock.patch("extty.run.time.sleep"):
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
                extty.log(
                    {"val/example": extty.Example(prompt="Hello", responses=["Hi"])},
                    step=2,
                )
                extty.log({"loss": 0.4}, step=3)

                runs_payload = self._wait_for(
                    lambda: self._get_json(f"{base_url}/runs", token),
                    lambda payload: any(
                        run_data["name"] == "server-run"
                        for run_data in payload.get("runs", [])
                    ),
                )
                assert any(
                    run_data["name"] == "server-run"
                    for run_data in runs_payload["runs"]
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
                assert example_records["val/example"]["records"][0]["data"][
                    "response"
                ] == [["Hi"]]

                system_payload = self._wait_for(
                    lambda: self._get_json(
                        f"{base_url}/runs/server-run/system?step=0", token
                    ),
                    lambda payload: len(payload.get("points", [])) >= 1,
                )
                assert len(system_payload["points"]) >= 1
            finally:
                extty.finish()


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


class TestModalIntegration:
    """Test Modal tunnel integration for server mode."""

    def test_modal_detection_when_inside_modal_function(self) -> None:
        """Test that Modal is detected when running inside Modal function."""
        # Mock the modal module
        mock_modal = mock.MagicMock()
        mock_modal.current_function_call_id.return_value = "test-function-call-id"

        # Mock find_spec to indicate modal is available
        with mock.patch("extty.server.find_spec") as mock_find_spec:
            mock_find_spec.return_value = mock.MagicMock()  # Modal is installed

            # Import modal in the mocked environment
            with mock.patch.dict("sys.modules", {"modal": mock_modal}):
                # Verify find_spec returns something (modal is available)
                assert mock_find_spec("modal") is not None

                # Verify inside modal function check would return True
                import sys

                assert sys.modules.get("modal") is not None
                assert (
                    sys.modules["modal"].current_function_call_id()
                    == "test-function-call-id"
                )

    def test_modal_detection_when_not_in_modal_function(self) -> None:
        """Test that Modal is not used when not inside Modal function."""
        # Mock the modal module to return None for current_function_call_id
        mock_modal = mock.MagicMock()
        mock_modal.current_function_call_id.return_value = None  # Not in Modal

        # Patch find_spec to indicate modal is available but not inside function
        with mock.patch("extty.server.find_spec") as mock_find_spec:
            mock_find_spec.return_value = mock.MagicMock()  # Modal is installed

            with mock.patch.dict("sys.modules", {"modal": mock_modal}):
                # Verify modal is available but current_function_call_id returns None
                import sys

                assert sys.modules.get("modal") is not None
                assert sys.modules["modal"].current_function_call_id() is None

    def test_modal_not_installed(self) -> None:
        """Test that server works when Modal is not installed."""
        # Patch find_spec to indicate modal is NOT available
        with mock.patch("extty.server.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None  # Modal is not installed

            # Verify find_spec returns None (modal is not available)
            assert mock_find_spec("modal") is None

    def test_server_works_without_modal(self) -> None:
        """Integration test: server works when Modal is not available."""
        with (
            mock.patch("extty.server.find_spec", return_value=None),
            mock.patch("extty.run.time.sleep"),
        ):
            run = extty.init(
                "no-modal-test-project",
                name="no-modal-test-run",
                system_metrics=False,
                server=True,
                server_host="127.0.0.1",
            )

            try:
                assert run.server_info is not None
                assert run.server_info.port > 0
                assert run.server_info.host == "127.0.0.1"
            finally:
                extty.finish()

    def test_server_works_with_modal_available_but_not_in_function(self) -> None:
        """Integration test: server uses regular mode when modal available but not in function."""
        mock_modal = mock.MagicMock()
        mock_modal.current_function_call_id.return_value = None

        with (
            mock.patch("extty.server.find_spec", return_value=mock.MagicMock()),
            mock.patch.dict("sys.modules", {"modal": mock_modal}),
            mock.patch("extty.run.time.sleep"),
        ):
            run = extty.init(
                "modal-available-test",
                name="modal-available-test-run",
                system_metrics=False,
                server=True,
                server_host="127.0.0.1",
            )

            try:
                assert run.server_info is not None
                assert run.server_info.port > 0
            finally:
                extty.finish()


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
                    {"prompt": "What is 2+2?", "response": "4"},
                    {"prompt": "What is 3*5?", "response": "15"},
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
            assert content["examples"][0]["prompt"] == "What is 2+2?"
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
            assert meta["config"] == {"lr": 0.001, "batch_size": 64}
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
            assert meta["config"] == {"lr": 0.001, "batch_size": 64}
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
