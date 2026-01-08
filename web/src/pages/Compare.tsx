import { useSearchParams, Link } from "react-router-dom";
import { useQueries } from "@tanstack/react-query";
import { fetchRun, fetchMetricPoints, type RunDetail, type MetricPoint } from "../api/client";
import { CompareChart } from "../components/CompareChart";

export function Compare() {
  const [searchParams] = useSearchParams();
  const idsParam = searchParams.get("ids") || "";
  const runIds = idsParam.split(",").map((id) => parseInt(id, 10)).filter((id) => id > 0);

  const runQueries = useQueries({
    queries: runIds.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => fetchRun(id),
    })),
  });

  const runs = runQueries
    .filter((q) => q.data)
    .map((q) => q.data as RunDetail);

  const allMetrics = [...new Set(runs.flatMap((r) => r.metrics))];

  if (runIds.length < 2) {
    return (
      <div>
        <p style={{ color: "#8b949e" }}>Select at least 2 runs to compare.</p>
        <Link to="/">← Back to runs</Link>
      </div>
    );
  }

  const isLoading = runQueries.some((q) => q.isLoading);

  if (isLoading) {
    return <p style={{ color: "#8b949e" }}>Loading...</p>;
  }

  return (
    <div>
      <div style={{ marginBottom: "8px" }}>
        <Link to="/" style={{ color: "#8b949e", fontSize: "14px" }}>← Back to runs</Link>
      </div>

      <h2 style={{ fontSize: "20px", fontWeight: 600, marginBottom: "24px" }}>
        Comparing {runs.length} runs
      </h2>

      <div style={{ marginBottom: "32px" }}>
        <h3 style={{ fontSize: "14px", color: "#8b949e", marginBottom: "12px" }}>Runs</h3>
        <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
          {runs.map((run, i) => (
            <div
              key={run.id}
              style={{
                padding: "12px 16px",
                background: "#161b22",
                borderRadius: "8px",
                borderLeft: `4px solid ${COLORS[i % COLORS.length]}`,
              }}
            >
              <div style={{ fontWeight: 500 }}>{run.name}</div>
              <div style={{ color: "#8b949e", fontSize: "13px" }}>{run.project_name}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: "32px" }}>
        <h3 style={{ fontSize: "14px", color: "#8b949e", marginBottom: "12px" }}>Config Diff</h3>
        <ConfigDiff runs={runs} />
      </div>

      <div>
        <h3 style={{ fontSize: "14px", color: "#8b949e", marginBottom: "16px" }}>Metrics</h3>
        {allMetrics.length === 0 ? (
          <p style={{ color: "#8b949e" }}>No metrics to compare</p>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(500px, 1fr))", gap: "24px" }}>
            {allMetrics.map((metricName) => (
              <CompareMetricChart key={metricName} runs={runs} metricName={metricName} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7", "#f85149"];

function ConfigDiff({ runs }: { runs: RunDetail[] }) {
  const allKeys = [...new Set(runs.flatMap((r) => Object.keys(r.config || {})))];

  if (allKeys.length === 0) {
    return <p style={{ color: "#8b949e" }}>No config to compare</p>;
  }

  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #30363d" }}>
          <th style={{ padding: "8px", textAlign: "left" }}>Key</th>
          {runs.map((run, i) => (
            <th
              key={run.id}
              style={{
                padding: "8px",
                textAlign: "left",
                borderLeft: `3px solid ${COLORS[i % COLORS.length]}`,
              }}
            >
              {run.name}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {allKeys.map((key) => {
          const values = runs.map((r) => r.config?.[key]);
          const allSame = values.every((v) => JSON.stringify(v) === JSON.stringify(values[0]));

          return (
            <tr key={key} style={{ borderBottom: "1px solid #21262d" }}>
              <td style={{ padding: "8px", color: "#8b949e" }}>{key}</td>
              {values.map((v, i) => (
                <td
                  key={i}
                  style={{
                    padding: "8px",
                    background: allSame ? "transparent" : "#21262d",
                  }}
                >
                  {v !== undefined ? String(v) : "-"}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function CompareMetricChart({ runs, metricName }: { runs: RunDetail[]; metricName: string }) {
  const queries = useQueries({
    queries: runs.map((run) => ({
      queryKey: ["metric", run.id, metricName],
      queryFn: () => fetchMetricPoints(run.id, metricName),
      enabled: run.metrics.includes(metricName),
    })),
  });

  const series = runs.map((run, i) => ({
    name: run.name,
    data: (queries[i].data || []) as MetricPoint[],
    color: COLORS[i % COLORS.length],
  }));

  return <CompareChart title={metricName} series={series} />;
}
