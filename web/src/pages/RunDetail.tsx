import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchRun, fetchMetricPoints } from "../api/client";
import { MetricChart } from "../components/MetricChart";

export function RunDetail() {
  const { id } = useParams<{ id: string }>();
  const runId = parseInt(id || "0", 10);

  const { data: run, isLoading, error } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => fetchRun(runId),
    enabled: runId > 0,
  });

  if (isLoading) {
    return <p style={{ color: "#8b949e" }}>Loading...</p>;
  }

  if (error || !run) {
    return <p style={{ color: "#f85149" }}>Run not found</p>;
  }

  const statusColors: Record<string, string> = {
    running: "#3fb950",
    completed: "#58a6ff",
    failed: "#f85149",
  };

  const duration = run.started_at && run.finished_at
    ? formatDuration(new Date(run.finished_at).getTime() - new Date(run.started_at).getTime())
    : run.started_at
      ? "Running..."
      : "-";

  return (
    <div>
      <div style={{ marginBottom: "8px" }}>
        <Link to="/" style={{ color: "#8b949e", fontSize: "14px" }}>← Back to runs</Link>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "20px", fontWeight: 600 }}>{run.name}</h2>
        <span style={{ color: "#8b949e" }}>({run.project_name})</span>
        <span
          style={{
            color: statusColors[run.status] || "#8b949e",
            fontWeight: 500,
            padding: "2px 8px",
            background: "#21262d",
            borderRadius: "12px",
            fontSize: "13px",
          }}
        >
          {run.status}
        </span>
        <span style={{ color: "#8b949e", fontSize: "14px" }}>{duration}</span>
      </div>

      {run.config && Object.keys(run.config).length > 0 && (
        <div style={{ marginBottom: "24px" }}>
          <h3 style={{ fontSize: "14px", color: "#8b949e", marginBottom: "8px" }}>Config</h3>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {Object.entries(run.config).map(([k, v]) => (
              <span
                key={k}
                style={{
                  padding: "4px 10px",
                  background: "#21262d",
                  borderRadius: "16px",
                  fontSize: "13px",
                }}
              >
                {k}={String(v)}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginBottom: "24px" }}>
        <h3 style={{ fontSize: "14px", color: "#8b949e", marginBottom: "16px" }}>Metrics</h3>
        {run.metrics.length === 0 ? (
          <p style={{ color: "#8b949e" }}>No metrics recorded</p>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: "24px" }}>
            {run.metrics.map((metricName) => (
              <MetricChartWrapper key={metricName} runId={runId} metricName={metricName} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricChartWrapper({ runId, metricName }: { runId: number; metricName: string }) {
  const { data: points = [] } = useQuery({
    queryKey: ["metric", runId, metricName],
    queryFn: () => fetchMetricPoints(runId, metricName),
  });

  return <MetricChart title={metricName} data={points} />;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}
