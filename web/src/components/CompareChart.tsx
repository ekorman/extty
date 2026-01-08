import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { MetricPoint } from "../api/client";

interface Series {
  name: string;
  data: MetricPoint[];
  color: string;
}

interface CompareChartProps {
  title: string;
  series: Series[];
}

export function CompareChart({ title, series }: CompareChartProps) {
  const allSteps = [...new Set(series.flatMap((s) => s.data.map((p) => p.step)))].sort(
    (a, b) => a - b
  );

  const chartData = allSteps.map((step) => {
    const point: Record<string, number> = { step };
    for (const s of series) {
      const p = s.data.find((d) => d.step === step);
      if (p) {
        point[s.name] = p.value;
      }
    }
    return point;
  });

  return (
    <div style={{ background: "#161b22", borderRadius: "8px", padding: "16px" }}>
      <h4 style={{ marginBottom: "12px", fontSize: "14px" }}>{title}</h4>
      <ResponsiveContainer width="100%" height={250}>
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
          <Legend />
          {series.map((s) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
