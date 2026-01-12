"""Tests for run endpoints."""

from fastapi.testclient import TestClient


def test_upload_run_minimal(client: TestClient):
    """Upload a run with minimal data."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    run = response.json()
    assert run["name"] == "test-run"
    assert run["status"] == "completed"
    assert run["project_id"] > 0
    assert run["config"] is None
    assert run["started_at"] is None
    assert run["finished_at"] is None


def test_upload_run_with_config(client: TestClient):
    """Upload a run with configuration."""
    data = {
        "project": "ml-project",
        "run_name": "experiment-1",
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam",
        },
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    run = response.json()
    assert run["config"] == {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
    }


def test_upload_run_with_timestamps(client: TestClient):
    """Upload a run with timestamps."""
    data = {
        "project": "test-project",
        "run_name": "timed-run",
        "started_at": "2024-01-01T10:00:00Z",
        "finished_at": "2024-01-01T11:30:00Z",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    run = response.json()
    assert run["started_at"] is not None
    assert run["finished_at"] is not None


def test_upload_run_duplicate(client: TestClient):
    """Uploading duplicate run should fail."""
    data = {
        "project": "test-project",
        "run_name": "duplicate-run",
        "status": "completed",
    }

    # First upload should succeed
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    # Second upload with same name should fail
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_upload_run_with_metrics(client: TestClient):
    """Upload a run with metrics."""
    data = {
        "project": "test-project",
        "run_name": "metrics-run",
        "status": "completed",
        "metrics": {
            "loss": [
                {"step": 0, "value": 1.5, "timestamp": 1000.0},
                {"step": 1, "value": 1.2, "timestamp": 1001.0},
                {"step": 2, "value": 0.9, "timestamp": 1002.0},
            ],
            "accuracy": [
                {"step": 0, "value": 0.5, "timestamp": 1000.0},
                {"step": 1, "value": 0.7, "timestamp": 1001.0},
            ],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    run_id = response.json()["id"]

    # Verify metrics were created
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    metric_names = response.json()
    assert set(metric_names) == {"loss", "accuracy"}


def test_upload_run_with_system_metrics(client: TestClient):
    """Upload a run with system metrics."""
    data = {
        "project": "test-project",
        "run_name": "system-run",
        "status": "completed",
        "system": [
            {
                "timestamp": 1000.0,
                "ram_used_gb": 4.5,
                "ram_total_gb": 16.0,
                "gpu_mem_used_gb": 2.0,
                "gpu_mem_total_gb": 8.0,
                "gpu_util_pct": 75.0,
            },
            {
                "timestamp": 1001.0,
                "ram_used_gb": 5.0,
                "ram_total_gb": 16.0,
                "gpu_mem_used_gb": 2.5,
                "gpu_mem_total_gb": 8.0,
                "gpu_util_pct": 80.0,
            },
        ],
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    run_id = response.json()["id"]

    # Verify system metrics were created
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    run_detail = response.json()
    assert len(run_detail["system_points"]) == 2
    assert run_detail["system_points"][0]["ram_used_gb"] == 4.5


def test_list_runs_empty(client: TestClient):
    """List runs should return empty list initially."""
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_list_runs_basic(client: TestClient):
    """List runs after creating some."""
    # Create a few runs
    for i in range(3):
        data = {
            "project": "test-project",
            "run_name": f"run-{i}",
            "status": "completed",
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # List all runs
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 3


def test_list_runs_filter_by_project(client: TestClient):
    """Filter runs by project."""
    # Create runs in different projects
    for project in ["project-a", "project-b"]:
        for i in range(2):
            data = {
                "project": project,
                "run_name": f"run-{i}",
                "status": "completed",
            }
            response = client.post("/api/v1/runs", json=data)
            assert response.status_code == 200

    # Filter by project-a
    response = client.get("/api/v1/runs?project=project-a")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 2
    assert all(run["project_name"] == "project-a" for run in runs)


def test_list_runs_filter_by_status(client: TestClient):
    """Filter runs by status."""
    # Create runs with different statuses
    for status in ["running", "completed", "failed"]:
        data = {
            "project": "test-project",
            "run_name": f"run-{status}",
            "status": status,
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # Filter by completed
    response = client.get("/api/v1/runs?status=completed")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"


def test_list_runs_search(client: TestClient):
    """Search runs by name or project."""
    # Create runs with different names
    data_items = [
        {"project": "ml-project", "run_name": "baseline"},
        {"project": "ml-project", "run_name": "experiment-1"},
        {"project": "vision-project", "run_name": "test-run"},
    ]

    for data in data_items:
        data["status"] = "completed"
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # Search for "experiment"
    response = client.get("/api/v1/runs?search=experiment")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["name"] == "experiment-1"

    # Search for "ml"
    response = client.get("/api/v1/runs?search=ml")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 2


def test_list_runs_sorting(client: TestClient):
    """Test sorting runs."""
    # Create runs with different names
    for name in ["zebra", "alpha", "beta"]:
        data = {
            "project": "test-project",
            "run_name": name,
            "status": "completed",
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # Sort by name ascending
    response = client.get("/api/v1/runs?sort=name&order=asc")
    assert response.status_code == 200
    runs = response.json()
    assert [run["name"] for run in runs] == ["alpha", "beta", "zebra"]

    # Sort by name descending
    response = client.get("/api/v1/runs?sort=name&order=desc")
    assert response.status_code == 200
    runs = response.json()
    assert [run["name"] for run in runs] == ["zebra", "beta", "alpha"]


def test_list_runs_pagination(client: TestClient):
    """Test pagination."""
    # Create 10 runs
    for i in range(10):
        data = {
            "project": "test-project",
            "run_name": f"run-{i:02d}",
            "status": "completed",
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # Get first 5
    response = client.get("/api/v1/runs?limit=5&offset=0")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 5

    # Get next 5
    response = client.get("/api/v1/runs?limit=5&offset=5")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 5


def test_get_run(client: TestClient):
    """Get a specific run."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "config": {"lr": 0.001},
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get the run
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200

    run = response.json()
    assert run["id"] == run_id
    assert run["name"] == "test-run"
    assert run["project_name"] == "test-project"
    assert run["config"] == {"lr": 0.001}
    assert "metrics" in run
    assert "system_points" in run


def test_get_run_not_found(client: TestClient):
    """Get non-existent run should return 404."""
    response = client.get("/api/v1/runs/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_delete_run(client: TestClient):
    """Delete a run."""
    data = {
        "project": "test-project",
        "run_name": "to-delete",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Delete the run
    response = client.delete(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted"}

    # Verify it's gone
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 404


def test_delete_run_not_found(client: TestClient):
    """Delete non-existent run should return 404."""
    response = client.delete("/api/v1/runs/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_delete_run_cascades(client: TestClient):
    """Deleting a run should delete associated metrics and system points."""
    data = {
        "project": "test-project",
        "run_name": "cascade-test",
        "status": "completed",
        "metrics": {
            "loss": [{"step": 0, "value": 1.0, "timestamp": 1000.0}],
        },
        "system": [
            {
                "timestamp": 1000.0,
                "ram_used_gb": 4.0,
                "ram_total_gb": 16.0,
                "gpu_mem_used_gb": None,
                "gpu_mem_total_gb": None,
                "gpu_util_pct": None,
            },
        ],
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Verify metrics exist
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    assert len(response.json()) == 1

    # Delete the run
    response = client.delete(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200

    # Verify metrics are gone (returns empty list for deleted run)
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    assert response.json() == []
