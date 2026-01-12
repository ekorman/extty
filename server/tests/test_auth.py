"""Tests for authentication."""

from fastapi.testclient import TestClient


def test_health_check_no_auth(client: TestClient):
    """Health check should work without authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_run_no_auth_required(client: TestClient):
    """Upload should work when API key is not set."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    response = client.post("/api/v1/runs", json=data)
    assert response.status_code == 200


def test_upload_run_with_auth_missing_token(auth_client: TestClient):
    """Upload should fail when API key is required but not provided."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    response = auth_client.post("/api/v1/runs", json=data)
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authorization header"


def test_upload_run_with_auth_invalid_token(auth_client: TestClient):
    """Upload should fail when API key is invalid."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    headers = {"Authorization": "Bearer wrong-key"}
    response = auth_client.post("/api/v1/runs", json=data, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_upload_run_with_auth_valid_token(auth_client: TestClient):
    """Upload should succeed with valid API key."""
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    headers = {"Authorization": "Bearer test-api-key-12345"}
    response = auth_client.post("/api/v1/runs", json=data, headers=headers)
    assert response.status_code == 200


def test_delete_run_with_auth_required(auth_client: TestClient):
    """Delete should require valid API key."""
    # First create a run
    data = {
        "project": "test-project",
        "run_name": "test-run",
        "status": "completed",
    }
    headers = {"Authorization": "Bearer test-api-key-12345"}
    response = auth_client.post("/api/v1/runs", json=data, headers=headers)
    assert response.status_code == 200
    run_id = response.json()["id"]

    # Try to delete without auth
    response = auth_client.delete(f"/api/v1/runs/{run_id}")
    assert response.status_code == 401

    # Try to delete with wrong auth
    headers = {"Authorization": "Bearer wrong-key"}
    response = auth_client.delete(f"/api/v1/runs/{run_id}", headers=headers)
    assert response.status_code == 401

    # Delete with correct auth
    headers = {"Authorization": "Bearer test-api-key-12345"}
    response = auth_client.delete(f"/api/v1/runs/{run_id}", headers=headers)
    assert response.status_code == 200


def test_read_endpoints_no_auth(auth_client: TestClient):
    """Read endpoints should not require authentication."""
    # List projects
    response = auth_client.get("/api/v1/projects")
    assert response.status_code == 200

    # List runs
    response = auth_client.get("/api/v1/runs")
    assert response.status_code == 200
