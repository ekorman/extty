import React from "react";
import { Box, Text } from "ink";
import * as asciichart from "asciichart";
import type { SystemPoint } from "../lib/parser.js";

interface SystemStatsProps {
  system: SystemPoint[];
  width?: number;
  height?: number;
}

export function SystemStats({
  system,
  width = 45,
  height = 8,
}: SystemStatsProps): React.ReactElement {
  if (system.length === 0) {
    return (
      <Box paddingX={1}>
        <Text dimColor>System metrics: waiting for data...</Text>
      </Box>
    );
  }

  const latest = system[system.length - 1];
  const ramPercent = (latest.ramUsedGb / latest.ramTotalGb) * 100;
  const hasGpu = latest.gpuMemUsedGb !== null && latest.gpuMemTotalGb !== null;

  const ramValues = system.map(
    (p) => (p.ramUsedGb / p.ramTotalGb) * 100
  );

  const gpuValues = hasGpu
    ? system.map((p) =>
        p.gpuUtilPct !== null ? p.gpuUtilPct : 0
      )
    : [];

  return (
    <Box flexDirection="row" gap={2} paddingX={1}>
      <SystemChart
        title="RAM %"
        values={ramValues}
        latest={ramPercent}
        unit="%"
        color="green"
        width={width}
        height={height}
        subtitle={`${latest.ramUsedGb.toFixed(1)}/${latest.ramTotalGb.toFixed(0)}GB`}
      />
      {hasGpu && gpuValues.length > 0 && (
        <SystemChart
          title="GPU Util %"
          values={gpuValues}
          latest={latest.gpuUtilPct ?? 0}
          unit="%"
          color="magenta"
          width={width}
          height={height}
          subtitle={`${latest.gpuMemUsedGb?.toFixed(1)}/${latest.gpuMemTotalGb?.toFixed(0)}GB VRAM`}
        />
      )}
    </Box>
  );
}

interface SystemChartProps {
  title: string;
  values: number[];
  latest: number;
  unit: string;
  color: string;
  width: number;
  height: number;
  subtitle?: string;
}

function SystemChart({
  title,
  values,
  latest,
  unit,
  color,
  width,
  height,
  subtitle,
}: SystemChartProps): React.ReactElement {
  const chartWidth = Math.max(width - 10, 10);
  const sampled = sampleData(values, chartWidth);

  const chart = asciichart.plot(sampled, {
    height: height - 3,
    format: (x: number) => x.toFixed(0).padStart(5),
    min: 0,
    max: 100,
  });

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={color}>
        {title}
      </Text>
      <Text>{chart}</Text>
      <Text dimColor>
        {latest.toFixed(1)}{unit} {subtitle && `| ${subtitle}`}
      </Text>
    </Box>
  );
}

function sampleData(values: number[], targetLength: number): number[] {
  if (values.length <= targetLength) return values;

  const result: number[] = [];
  const step = values.length / targetLength;

  for (let i = 0; i < targetLength; i++) {
    const idx = Math.floor(i * step);
    result.push(values[idx]);
  }

  return result;
}
