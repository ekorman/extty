"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ProjectSchema(BaseModel):
    id: int
    name: str
    created_at: datetime


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


class MetricPointSchema(BaseModel):
    step: int
    value: float
    timestamp: float


class MetricSchema(BaseModel):
    name: str
    points: list[MetricPointSchema] = []


class SystemPointSchema(BaseModel):
    timestamp: float
    ram_used_gb: float | None
    ram_total_gb: float | None
    gpu_mem_used_gb: float | None
    gpu_mem_total_gb: float | None
    gpu_util_pct: float | None


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
