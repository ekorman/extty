"""Runs API endpoints."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import verify_api_key
from app.database import get_db, encode_json, decode_json
from app.models import (
    RunSchema,
    RunDetailSchema,
    RunUploadSchema,
    SystemPointSchema,
)

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunSchema, dependencies=[Depends(verify_api_key)])
def upload_run(
    data: RunUploadSchema,
    conn: Annotated[object, Depends(get_db)],
) -> RunSchema:
    """Upload a run from the Python client."""
    # Get or create project
    cursor = conn.execute("SELECT id FROM projects WHERE name = ?", (data.project,))
    project_row = cursor.fetchone()

    if project_row:
        project_id = project_row[0]
    else:
        cursor = conn.execute("INSERT INTO projects (name) VALUES (?)", (data.project,))
        project_id = cursor.lastrowid

    # Check if run already exists
    cursor = conn.execute(
        "SELECT id FROM runs WHERE project_id = ? AND name = ?",
        (project_id, data.run_name),
    )
    if cursor.fetchone():
        raise HTTPException(
            status_code=409,
            detail=f"Run '{data.run_name}' already exists in project '{data.project}'",
        )

    # Parse timestamps
    started_at = None
    if data.started_at:
        started_at = datetime.fromisoformat(data.started_at.replace("Z", "+00:00"))

    finished_at = None
    if data.finished_at:
        finished_at = datetime.fromisoformat(data.finished_at.replace("Z", "+00:00"))

    # Insert run
    cursor = conn.execute(
        """
        INSERT INTO runs (project_id, name, config, started_at, finished_at, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            data.run_name,
            encode_json(data.config),
            started_at,
            finished_at,
            data.status,
        ),
    )
    run_id = cursor.lastrowid

    # Insert metrics
    for metric_name, points in data.metrics.items():
        cursor = conn.execute(
            "INSERT INTO metrics (run_id, name) VALUES (?, ?)",
            (run_id, metric_name),
        )
        metric_id = cursor.lastrowid

        for p in points:
            conn.execute(
                "INSERT INTO metric_points (metric_id, step, value, timestamp) VALUES (?, ?, ?, ?)",
                (metric_id, p.step, p.value, p.timestamp),
            )

    # Insert system points
    for sp in data.system:
        conn.execute(
            """
            INSERT INTO system_points
            (run_id, timestamp, ram_used_gb, ram_total_gb, gpu_mem_used_gb, gpu_mem_total_gb, gpu_util_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sp.timestamp,
                sp.ram_used_gb,
                sp.ram_total_gb,
                sp.gpu_mem_used_gb,
                sp.gpu_mem_total_gb,
                sp.gpu_util_pct,
            ),
        )

    # Fetch and return the created run
    cursor = conn.execute(
        "SELECT id, project_id, name, config, started_at, finished_at, status, created_at FROM runs WHERE id = ?",
        (run_id,),
    )
    run_row = cursor.fetchone()
    run_dict = dict(run_row)
    run_dict["config"] = decode_json(run_dict["config"])
    run_dict["project_name"] = data.project

    return RunSchema(**run_dict)


@router.get("", response_model=list[RunSchema])
def list_runs(
    conn: Annotated[object, Depends(get_db)],
    project: str | None = None,
    status: str | None = None,
    search: str | None = None,
    sort: str = "created_at",
    order: str = "desc",
    limit: int = Query(default=50, le=200),
    offset: int = 0,
) -> list[RunSchema]:
    """List runs with optional filters."""
    # Build query dynamically
    query_parts = [
        "SELECT r.*, p.name as project_name FROM runs r JOIN projects p ON r.project_id = p.id"
    ]
    params = []
    where_clauses = []

    if project:
        where_clauses.append("p.name = ?")
        params.append(project)

    if status:
        where_clauses.append("r.status = ?")
        params.append(status)

    if search:
        where_clauses.append("(r.name LIKE ? OR p.name LIKE ?)")
        search_pattern = f"%{search}%"
        params.extend([search_pattern, search_pattern])

    if where_clauses:
        query_parts.append("WHERE " + " AND ".join(where_clauses))

    # Add sorting
    valid_sort_columns = ["name", "created_at", "started_at", "finished_at", "status"]
    sort_column = sort if sort in valid_sort_columns else "created_at"
    order_dir = "DESC" if order == "desc" else "ASC"
    query_parts.append(f"ORDER BY r.{sort_column} {order_dir}")

    # Add pagination
    query_parts.append("LIMIT ? OFFSET ?")
    params.extend([limit, offset])

    query = " ".join(query_parts)
    cursor = conn.execute(query, tuple(params))

    runs = []
    for row in cursor.fetchall():
        run_dict = dict(row)
        run_dict["config"] = decode_json(run_dict["config"])
        runs.append(RunSchema(**run_dict))

    return runs


@router.get("/{run_id}", response_model=RunDetailSchema)
def get_run(
    run_id: int,
    conn: Annotated[object, Depends(get_db)],
) -> RunDetailSchema:
    """Get run details including metric names and system points."""
    # Get run with project name
    cursor = conn.execute(
        """
        SELECT r.*, p.name as project_name
        FROM runs r
        JOIN projects p ON r.project_id = p.id
        WHERE r.id = ?
        """,
        (run_id,),
    )
    run_row = cursor.fetchone()

    if not run_row:
        raise HTTPException(status_code=404, detail="Run not found")

    run_dict = dict(run_row)
    run_dict["config"] = decode_json(run_dict["config"])

    # Get metric names
    cursor = conn.execute("SELECT name FROM metrics WHERE run_id = ?", (run_id,))
    metric_names = [row[0] for row in cursor.fetchall()]

    # Get system points
    cursor = conn.execute(
        """
        SELECT timestamp, ram_used_gb, ram_total_gb, gpu_mem_used_gb, gpu_mem_total_gb, gpu_util_pct
        FROM system_points
        WHERE run_id = ?
        ORDER BY timestamp
        """,
        (run_id,),
    )
    system_points = [SystemPointSchema(**dict(row)) for row in cursor.fetchall()]

    run_dict["metrics"] = metric_names
    run_dict["system_points"] = system_points

    return RunDetailSchema(**run_dict)


@router.delete("/{run_id}", dependencies=[Depends(verify_api_key)])
def delete_run(
    run_id: int,
    conn: Annotated[object, Depends(get_db)],
) -> dict[str, str]:
    """Delete a run."""
    cursor = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,))

    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Run not found")

    conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    return {"status": "deleted"}


@router.get("/{run_id}/metrics", response_model=list[str])
def list_metrics(
    run_id: int,
    conn: Annotated[object, Depends(get_db)],
) -> list[str]:
    """List metric names for a run."""
    cursor = conn.execute("SELECT name FROM metrics WHERE run_id = ?", (run_id,))
    metric_names = [row[0] for row in cursor.fetchall()]
    return metric_names


@router.get("/{run_id}/metrics/{metric_name}")
def get_metric_points(
    run_id: int,
    metric_name: str,
    conn: Annotated[object, Depends(get_db)],
) -> list[dict]:
    """Get data points for a specific metric."""
    # Get metric ID
    cursor = conn.execute(
        "SELECT id FROM metrics WHERE run_id = ? AND name = ?",
        (run_id, metric_name),
    )
    metric_row = cursor.fetchone()

    if not metric_row:
        raise HTTPException(status_code=404, detail="Metric not found")

    metric_id = metric_row[0]

    # Get metric points
    cursor = conn.execute(
        """
        SELECT step, value, timestamp
        FROM metric_points
        WHERE metric_id = ?
        ORDER BY step
        """,
        (metric_id,),
    )

    points = [
        {"step": row[0], "value": row[1], "timestamp": row[2]}
        for row in cursor.fetchall()
    ]
    return points
