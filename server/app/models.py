"""SQLAlchemy models and Pydantic schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import ForeignKey, Index, JSON, String, Float, Integer, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="project")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="runs")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    system_points: Mapped[list["SystemPoint"]] = relationship(back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_runs_project", "project_id"),
    )


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)

    run: Mapped["Run"] = relationship(back_populates="metrics")
    points: Mapped[list["MetricPoint"]] = relationship(back_populates="metric", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_metrics_run_name", "run_id", "name", unique=True),
    )


class MetricPoint(Base):
    __tablename__ = "metric_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    metric_id: Mapped[int] = mapped_column(ForeignKey("metrics.id"))
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)

    metric: Mapped["Metric"] = relationship(back_populates="points")

    __table_args__ = (
        Index("idx_metric_points_metric", "metric_id"),
        Index("idx_metric_points_step", "metric_id", "step"),
    )


class SystemPoint(Base):
    __tablename__ = "system_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    ram_used_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    ram_total_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_mem_used_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_mem_total_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_util_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="system_points")

    __table_args__ = (
        Index("idx_system_points_run", "run_id"),
    )


class ProjectSchema(BaseModel):
    id: int
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RunSchema(BaseModel):
    id: int
    project_id: int
    project_name: str | None = None
    name: str
    config: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MetricPointSchema(BaseModel):
    step: int
    value: float
    timestamp: float

    model_config = {"from_attributes": True}


class MetricSchema(BaseModel):
    name: str
    points: list[MetricPointSchema] = []

    model_config = {"from_attributes": True}


class SystemPointSchema(BaseModel):
    timestamp: float
    ram_used_gb: float | None
    ram_total_gb: float | None
    gpu_mem_used_gb: float | None
    gpu_mem_total_gb: float | None
    gpu_util_pct: float | None

    model_config = {"from_attributes": True}


class RunDetailSchema(RunSchema):
    metrics: list[str] = []
    system_points: list[SystemPointSchema] = []


class RunUploadSchema(BaseModel):
    project: str
    run_name: str
    config: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "completed"
    metrics: dict[str, list[MetricPointSchema]] = {}
    system: list[SystemPointSchema] = []
