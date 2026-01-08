"""Runs API endpoints."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, or_
from sqlalchemy.orm import Session, selectinload

from app.database import get_db
from app.models import (
    Project,
    Run,
    Metric,
    MetricPoint,
    SystemPoint,
    RunSchema,
    RunDetailSchema,
    RunUploadSchema,
    MetricPointSchema,
    SystemPointSchema,
)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunSchema)
def upload_run(
    data: RunUploadSchema,
    db: Annotated[Session, Depends(get_db)],
) -> Run:
    """Upload a run from the Python client."""
    project = db.execute(
        select(Project).where(Project.name == data.project)
    ).scalar_one_or_none()

    if not project:
        project = Project(name=data.project)
        db.add(project)
        db.flush()

    existing = db.execute(
        select(Run).where(Run.project_id == project.id, Run.name == data.run_name)
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{data.run_name}' already exists in project '{data.project}'",
        )

    started_at = None
    if data.started_at:
        started_at = datetime.fromisoformat(data.started_at.replace("Z", "+00:00"))

    finished_at = None
    if data.finished_at:
        finished_at = datetime.fromisoformat(data.finished_at.replace("Z", "+00:00"))

    run = Run(
        project_id=project.id,
        name=data.run_name,
        config=data.config,
        started_at=started_at,
        finished_at=finished_at,
        status=data.status,
    )
    db.add(run)
    db.flush()

    for metric_name, points in data.metrics.items():
        metric = Metric(run_id=run.id, name=metric_name)
        db.add(metric)
        db.flush()

        for p in points:
            db.add(MetricPoint(
                metric_id=metric.id,
                step=p.step,
                value=p.value,
                timestamp=p.timestamp,
            ))

    for sp in data.system:
        db.add(SystemPoint(
            run_id=run.id,
            timestamp=sp.timestamp,
            ram_used_gb=sp.ram_used_gb,
            ram_total_gb=sp.ram_total_gb,
            gpu_mem_used_gb=sp.gpu_mem_used_gb,
            gpu_mem_total_gb=sp.gpu_mem_total_gb,
            gpu_util_pct=sp.gpu_util_pct,
        ))

    db.commit()
    db.refresh(run)

    return run


@router.get("", response_model=list[RunSchema])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    project: str | None = None,
    status: str | None = None,
    search: str | None = None,
    sort: str = "created_at",
    order: str = "desc",
    limit: int = Query(default=50, le=200),
    offset: int = 0,
) -> list[Run]:
    """List runs with optional filters."""
    query = select(Run).options(selectinload(Run.project))

    if project:
        query = query.join(Project).where(Project.name == project)

    if status:
        query = query.where(Run.status == status)

    if search:
        query = query.where(
            or_(
                Run.name.ilike(f"%{search}%"),
                Run.project.has(Project.name.ilike(f"%{search}%")),
            )
        )

    sort_column = getattr(Run, sort, Run.created_at)
    if order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    query = query.offset(offset).limit(limit)

    runs = db.execute(query).scalars().all()

    for run in runs:
        run.project_name = run.project.name

    return list(runs)


@router.get("/{run_id}", response_model=RunDetailSchema)
def get_run(
    run_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> RunDetailSchema:
    """Get run details including metric names and system points."""
    run = db.execute(
        select(Run)
        .options(
            selectinload(Run.project),
            selectinload(Run.metrics),
            selectinload(Run.system_points),
        )
        .where(Run.id == run_id)
    ).scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunDetailSchema(
        id=run.id,
        project_id=run.project_id,
        project_name=run.project.name,
        name=run.name,
        config=run.config,
        started_at=run.started_at,
        finished_at=run.finished_at,
        status=run.status,
        created_at=run.created_at,
        metrics=[m.name for m in run.metrics],
        system_points=[
            SystemPointSchema(
                timestamp=sp.timestamp,
                ram_used_gb=sp.ram_used_gb,
                ram_total_gb=sp.ram_total_gb,
                gpu_mem_used_gb=sp.gpu_mem_used_gb,
                gpu_mem_total_gb=sp.gpu_mem_total_gb,
                gpu_util_pct=sp.gpu_util_pct,
            )
            for sp in run.system_points
        ],
    )


@router.delete("/{run_id}")
def delete_run(
    run_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Delete a run."""
    run = db.execute(select(Run).where(Run.id == run_id)).scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    db.delete(run)
    db.commit()

    return {"status": "deleted"}


@router.get("/{run_id}/metrics", response_model=list[str])
def list_metrics(
    run_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> list[str]:
    """List metric names for a run."""
    metrics = db.execute(
        select(Metric.name).where(Metric.run_id == run_id)
    ).scalars().all()

    return list(metrics)


@router.get("/{run_id}/metrics/{metric_name}", response_model=list[MetricPointSchema])
def get_metric_points(
    run_id: int,
    metric_name: str,
    db: Annotated[Session, Depends(get_db)],
) -> list[MetricPointSchema]:
    """Get data points for a specific metric."""
    metric = db.execute(
        select(Metric)
        .options(selectinload(Metric.points))
        .where(Metric.run_id == run_id, Metric.name == metric_name)
    ).scalar_one_or_none()

    if not metric:
        raise HTTPException(status_code=404, detail="Metric not found")

    return [
        MetricPointSchema(step=p.step, value=p.value, timestamp=p.timestamp)
        for p in sorted(metric.points, key=lambda x: x.step)
    ]
