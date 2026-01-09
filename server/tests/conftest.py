"""Test configuration and fixtures."""

import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base, get_db
from app.main import app


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """Create a test database for each test."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create a session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture(scope="function")
def client(test_db: Session) -> TestClient:
    """Create a test client with a test database."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def auth_client(test_db: Session, monkeypatch) -> TestClient:
    """Create a test client with authentication enabled."""
    # Set API key for authentication tests
    monkeypatch.setenv("EXTTY_API_KEY", "test-api-key-12345")

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
