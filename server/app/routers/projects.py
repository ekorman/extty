"""Projects API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db, decode_json
from app.models import ProjectSchema, RunSchema

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSchema])
def list_projects(
    conn: Annotated[object, Depends(get_db)],
) -> list[ProjectSchema]:
    """List all projects."""
    cursor = conn.execute("SELECT id, name, created_at FROM projects ORDER BY name")
    projects = [ProjectSchema(**dict(row)) for row in cursor.fetchall()]
    return projects


@router.get("/{project_name}", response_model=list[RunSchema])
def get_project_runs(
    project_name: str,
    conn: Annotated[object, Depends(get_db)],
) -> list[RunSchema]:
    """Get all runs for a project."""
    # Check if project exists
    cursor = conn.execute(
        "SELECT id, name FROM projects WHERE name = ?", (project_name,)
    )
    project = cursor.fetchone()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_id = project[0]
    project_name_actual = project[1]

    # Get all runs for the project
    cursor = conn.execute(
        """
        SELECT id, project_id, name, config, started_at, finished_at, status, created_at
        FROM runs
        WHERE project_id = ?
        ORDER BY created_at DESC
        """,
        (project_id,),
    )

    runs = []
    for row in cursor.fetchall():
        run_dict = dict(row)
        # Decode JSON config
        run_dict["config"] = decode_json(run_dict["config"])
        run_dict["project_name"] = project_name_actual
        runs.append(RunSchema(**run_dict))

    return runs
