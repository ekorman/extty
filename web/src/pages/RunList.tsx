import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { fetchRuns, fetchProjects, type Run } from "../api/client";

export function RunList() {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [projectFilter, setProjectFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["runs", { search, project: projectFilter, status: statusFilter }],
    queryFn: () => fetchRuns({ search, project: projectFilter, status: statusFilter }),
  });

  const { data: projects = [] } = useQuery({
    queryKey: ["projects"],
    queryFn: fetchProjects,
  });

  const toggleSelect = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleCompare = () => {
    if (selected.size >= 2) {
      navigate(`/compare?ids=${Array.from(selected).join(",")}`);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", gap: "12px", marginBottom: "20px", flexWrap: "wrap" }}>
        <input
          type="text"
          placeholder="Search runs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "6px",
            color: "#e6edf3",
            width: "200px",
          }}
        />
        <select
          value={projectFilter}
          onChange={(e) => setProjectFilter(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "6px",
            color: "#e6edf3",
          }}
        >
          <option value="">All projects</option>
          {projects.map((p) => (
            <option key={p.id} value={p.name}>{p.name}</option>
          ))}
        </select>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#161b22",
            border: "1px solid #30363d",
            borderRadius: "6px",
            color: "#e6edf3",
          }}
        >
          <option value="">All statuses</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
        {selected.size >= 2 && (
          <button
            onClick={handleCompare}
            style={{
              padding: "8px 16px",
              background: "#238636",
              border: "none",
              borderRadius: "6px",
              color: "#fff",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            Compare ({selected.size})
          </button>
        )}
      </div>

      {isLoading ? (
        <p style={{ color: "#8b949e" }}>Loading...</p>
      ) : runs.length === 0 ? (
        <p style={{ color: "#8b949e" }}>No runs found. Push a run with `extty push`.</p>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #30363d" }}>
              <th style={{ padding: "12px 8px", textAlign: "left", width: "40px" }}></th>
              <th style={{ padding: "12px 8px", textAlign: "left" }}>Name</th>
              <th style={{ padding: "12px 8px", textAlign: "left" }}>Project</th>
              <th style={{ padding: "12px 8px", textAlign: "left" }}>Status</th>
              <th style={{ padding: "12px 8px", textAlign: "left" }}>Started</th>
              <th style={{ padding: "12px 8px", textAlign: "left" }}>Config</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <RunRow
                key={run.id}
                run={run}
                selected={selected.has(run.id)}
                onToggleSelect={() => toggleSelect(run.id)}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function RunRow({
  run,
  selected,
  onToggleSelect,
}: {
  run: Run;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  const statusColors: Record<string, string> = {
    running: "#3fb950",
    completed: "#58a6ff",
    failed: "#f85149",
  };

  const configPreview = run.config
    ? Object.entries(run.config)
        .slice(0, 3)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ")
    : "-";

  return (
    <tr
      style={{
        borderBottom: "1px solid #21262d",
        background: selected ? "#161b22" : "transparent",
      }}
    >
      <td style={{ padding: "12px 8px" }}>
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggleSelect}
          style={{ cursor: "pointer" }}
        />
      </td>
      <td style={{ padding: "12px 8px" }}>
        <Link to={`/runs/${run.id}`} style={{ fontWeight: 500 }}>
          {run.name}
        </Link>
      </td>
      <td style={{ padding: "12px 8px", color: "#8b949e" }}>{run.project_name}</td>
      <td style={{ padding: "12px 8px" }}>
        <span
          style={{
            color: statusColors[run.status] || "#8b949e",
            fontWeight: 500,
          }}
        >
          {run.status}
        </span>
      </td>
      <td style={{ padding: "12px 8px", color: "#8b949e" }}>
        {run.started_at ? new Date(run.started_at).toLocaleString() : "-"}
      </td>
      <td style={{ padding: "12px 8px", color: "#8b949e", fontSize: "13px" }}>
        {configPreview}
      </td>
    </tr>
  );
}
