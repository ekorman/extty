"""Database connection and session management."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_PATH = Path.home() / ".extty" / "server.db"


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DATABASE_PATH}"


engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
