"""Tests for metrics endpoints."""

from fastapi.testclient import TestClient


def test_list_metrics_empty(client: TestClient):
    """List metrics for run without metrics."""
    data = {
        "project": "test-project",
        "run_name": "no-metrics",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # List metrics
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    assert response.json() == []


def test_list_metrics_single(client: TestClient):
    """List metrics for run with one metric."""
    data = {
        "project": "test-project",
        "run_name": "single-metric",
        "status": "completed",
        "metrics": {
            "loss": [
                {"step": 0, "value": 1.0, "timestamp": 1000.0},
            ],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # List metrics
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics == ["loss"]


def test_list_metrics_multiple(client: TestClient):
    """List multiple metrics for a run."""
    data = {
        "project": "test-project",
        "run_name": "multi-metric",
        "status": "completed",
        "metrics": {
            "loss": [{"step": 0, "value": 1.0, "timestamp": 1000.0}],
            "accuracy": [{"step": 0, "value": 0.5, "timestamp": 1000.0}],
            "f1_score": [{"step": 0, "value": 0.6, "timestamp": 1000.0}],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # List metrics
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert set(metrics) == {"loss", "accuracy", "f1_score"}


def test_get_metric_points(client: TestClient):
    """Get data points for a metric."""
    data = {
        "project": "test-project",
        "run_name": "metric-points",
        "status": "completed",
        "metrics": {
            "loss": [
                {"step": 0, "value": 1.5, "timestamp": 1000.0},
                {"step": 1, "value": 1.2, "timestamp": 1001.0},
                {"step": 2, "value": 0.9, "timestamp": 1002.0},
            ],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get metric points
    response = client.get(f"/api/v1/runs/{run_id}/metrics/loss")
    assert response.status_code == 200
    points = response.json()

    assert len(points) == 3
    assert points[0] == {"step": 0, "value": 1.5, "timestamp": 1000.0}
    assert points[1] == {"step": 1, "value": 1.2, "timestamp": 1001.0}
    assert points[2] == {"step": 2, "value": 0.9, "timestamp": 1002.0}


def test_get_metric_points_sorted_by_step(client: TestClient):
    """Metric points should be sorted by step."""
    data = {
        "project": "test-project",
        "run_name": "unsorted-points",
        "status": "completed",
        "metrics": {
            "loss": [
                {"step": 2, "value": 0.9, "timestamp": 1002.0},
                {"step": 0, "value": 1.5, "timestamp": 1000.0},
                {"step": 1, "value": 1.2, "timestamp": 1001.0},
            ],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get metric points
    response = client.get(f"/api/v1/runs/{run_id}/metrics/loss")
    assert response.status_code == 200
    points = response.json()

    # Should be sorted by step
    assert [p["step"] for p in points] == [0, 1, 2]


def test_get_metric_points_not_found(client: TestClient):
    """Get points for non-existent metric."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
        "metrics": {
            "loss": [{"step": 0, "value": 1.0, "timestamp": 1000.0}],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Try to get non-existent metric
    response = client.get(f"/api/v1/runs/{run_id}/metrics/accuracy")
    assert response.status_code == 404
    assert response.json()["detail"] == "Metric not found"


def test_metric_with_special_characters(client: TestClient):
    """Test metrics with special characters in names can be stored and listed.

    Note: Due to FastAPI path parameter limitations, metrics with slashes
    cannot be retrieved via the /{run_id}/metrics/{metric_name} endpoint
    without using a path converter. They can still be stored and listed.
    """
    data = {
        "project": "test-project",
        "run_name": "special-metrics",
        "status": "completed",
        "metrics": {
            "train/loss": [{"step": 0, "value": 1.0, "timestamp": 1000.0}],
            "val/accuracy": [{"step": 0, "value": 0.8, "timestamp": 1000.0}],
            "train_accuracy": [{"step": 0, "value": 0.9, "timestamp": 1000.0}],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # List metrics - metrics with slashes should be stored and listed correctly
    response = client.get(f"/api/v1/runs/{run_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert set(metrics) == {"train/loss", "val/accuracy", "train_accuracy"}

    # Metrics without slashes can be retrieved normally
    response = client.get(f"/api/v1/runs/{run_id}/metrics/train_accuracy")
    assert response.status_code == 200
    points = response.json()
    assert len(points) == 1
    assert points[0]["value"] == 0.9


def test_metric_with_many_points(client: TestClient):
    """Test metric with many data points."""
    # Create 100 metric points
    points = [
        {"step": i, "value": 1.0 / (i + 1), "timestamp": 1000.0 + i} for i in range(100)
    ]

    data = {
        "project": "test-project",
        "run_name": "many-points",
        "status": "completed",
        "metrics": {
            "loss": points,
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get all points
    response = client.get(f"/api/v1/runs/{run_id}/metrics/loss")
    assert response.status_code == 200
    retrieved_points = response.json()
    assert len(retrieved_points) == 100


def test_multiple_metrics_different_lengths(client: TestClient):
    """Test run with metrics of different lengths."""
    data = {
        "project": "test-project",
        "run_name": "diff-lengths",
        "status": "completed",
        "metrics": {
            "loss": [
                {"step": i, "value": 1.0 - i * 0.1, "timestamp": 1000.0 + i}
                for i in range(10)
            ],
            "accuracy": [
                {"step": i * 2, "value": i * 0.05, "timestamp": 1000.0 + i * 2}
                for i in range(5)
            ],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get loss (10 points)
    response = client.get(f"/api/v1/runs/{run_id}/metrics/loss")
    assert response.status_code == 200
    assert len(response.json()) == 10

    # Get accuracy (5 points)
    response = client.get(f"/api/v1/runs/{run_id}/metrics/accuracy")
    assert response.status_code == 200
    assert len(response.json()) == 5


def test_run_detail_includes_metric_names(client: TestClient):
    """Run detail should include metric names."""
    data = {
        "project": "test-project",
        "run_name": "detail-test",
        "status": "completed",
        "metrics": {
            "loss": [{"step": 0, "value": 1.0, "timestamp": 1000.0}],
            "accuracy": [{"step": 0, "value": 0.8, "timestamp": 1000.0}],
        },
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Get run detail
    response = client.get(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200
    run = response.json()

    assert "metrics" in run
    assert set(run["metrics"]) == {"loss", "accuracy"}
