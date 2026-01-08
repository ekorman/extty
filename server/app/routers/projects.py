"""Projects API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.database import get_db
from app.models import Project, Run, ProjectSchema, RunSchema

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSchema])
def list_projects(
    db: Annotated[Session, Depends(get_db)],
) -> list[Project]:
    """List all projects."""
    projects = db.execute(
        select(Project).order_by(Project.name)
    ).scalars().all()

    return list(projects)


@router.get("/{project_name}", response_model=list[RunSchema])
def get_project_runs(
    project_name: str,
    db: Annotated[Session, Depends(get_db)],
) -> list[Run]:
    """Get all runs for a project."""
    project = db.execute(
        select(Project)
        .options(selectinload(Project.runs))
        .where(Project.name == project_name)
    ).scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    for run in project.runs:
        run.project_name = project.name

    return list(project.runs)
