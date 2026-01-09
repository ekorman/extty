"""Tests for project endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_list_projects_empty(client: TestClient):
    """List projects should return empty list initially."""
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    assert response.json() == []


def test_list_projects_after_run_creation(client: TestClient):
    """Projects should appear after creating runs."""
    # Create a run which creates a project
    data = {
        "project": "my-project",
        "run_name": "run-1",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    # List projects
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) == 1
    assert projects[0]["name"] == "my-project"
    assert "id" in projects[0]
    assert "created_at" in projects[0]


def test_list_projects_multiple(client: TestClient):
    """List multiple projects sorted by name."""
    # Create runs in different projects
    for project_name in ["zebra", "alpha", "beta"]:
        data = {
            "project": project_name,
            "run_name": "run-1",
            "status": "completed",
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # List projects (should be sorted alphabetically)
    response = client.get("/api/v1/projects")
    assert response.status_code == 200
    projects = response.json()
    assert len(projects) == 3
    assert [p["name"] for p in projects] == ["alpha", "beta", "zebra"]


def test_get_project_runs_not_found(client: TestClient):
    """Getting runs for non-existent project should return 404."""
    response = client.get("/api/v1/projects/non-existent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_get_project_runs_empty(client: TestClient):
    """Getting runs for project with no runs should return empty list."""
    # Create a project with a run, then delete the run
    data = {
        "project": "empty-project",
        "run_name": "temp-run",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    run_id = response.json()["id"]

    # Delete the run
    response = client.delete(f"/api/v1/runs/{run_id}")
    assert response.status_code == 200

    # Get project runs
    response = client.get("/api/v1/projects/empty-project")
    assert response.status_code == 200
    assert response.json() == []


def test_get_project_runs_single(client: TestClient):
    """Get runs for a project with one run."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "config": {"lr": 0.001},
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200

    # Get project runs
    response = client.get("/api/v1/projects/test-project")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 1
    assert runs[0]["name"] == "test-run"
    assert runs[0]["project_name"] == "test-project"
    assert runs[0]["config"] == {"lr": 0.001}
    assert runs[0]["status"] == "completed"


def test_get_project_runs_multiple(client: TestClient):
    """Get multiple runs for a project."""
    project_name = "multi-run-project"

    # Create multiple runs
    for i in range(3):
        data = {
            "project": project_name,
            "run_name": f"run-{i}",
            "status": "completed",
        }
        response = client.post("/api/v1/runs", json=data)
        assert response.status_code == 200

    # Get project runs
    response = client.get(f"/api/v1/projects/{project_name}")
    assert response.status_code == 200
    runs = response.json()
    assert len(runs) == 3
    assert all(run["project_name"] == project_name for run in runs)
    assert {run["name"] for run in runs} == {"run-0", "run-1", "run-2"}
