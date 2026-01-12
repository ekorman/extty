"""Test configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.database import Database, init_db, get_db
from app.main import app


@pytest.fixture(scope="function")
def test_db() -> Generator[Database, None, None]:
    """Create a test database for each test."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")

    try:
        # Create database instance with temp path
        db = Database(db_path=Path(db_path))
        init_db(db)
        yield db
    finally:
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture(scope="function")
def client(test_db: Database) -> TestClient:
    """Create a test client with a test database."""

    def override_get_db():
        with test_db.get_connection() as conn:
            yield conn

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def auth_client(test_db: Database, monkeypatch) -> TestClient:
    """Create a test client with authentication enabled."""
    # Set API key for authentication tests
    monkeypatch.setenv("EXTTY_API_KEY", "test-api-key-12345")

    def override_get_db():
        with test_db.get_connection() as conn:
            yield conn

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
