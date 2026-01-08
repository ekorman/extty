import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { MetricPoint } from "../api/client";

interface MetricChartProps {
  title: string;
  data: MetricPoint[];
}

export function MetricChart({ title, data }: MetricChartProps) {
  if (data.length === 0) {
    return (
      <div style={{ background: "#161b22", borderRadius: "8px", padding: "16px" }}>
        <h4 style={{ marginBottom: "12px", fontSize: "14px" }}>{title}</h4>
        <p style={{ color: "#8b949e" }}>No data</p>
      </div>
    );
  }

  const chartData = data.map((p) => ({
    step: p.step,
    value: p.value,
  }));

  return (
    <div style={{ background: "#161b22", borderRadius: "8px", padding: "16px" }}>
      <h4 style={{ marginBottom: "12px", fontSize: "14px" }}>{title}</h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
          <XAxis
            dataKey="step"
            stroke="#8b949e"
            tick={{ fill: "#8b949e", fontSize: 11 }}
          />
          <YAxis
            stroke="#8b949e"
            tick={{ fill: "#8b949e", fontSize: 11 }}
            tickFormatter={(v) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{
              background: "#21262d",
              border: "1px solid #30363d",
              borderRadius: "6px",
            }}
            labelStyle={{ color: "#8b949e" }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#58a6ff"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
