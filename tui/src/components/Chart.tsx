import React from "react";
import { Box, Text } from "ink";
import * as asciichart from "asciichart";
import type { MetricPoint } from "../lib/parser.js";

interface ChartProps {
  title: string;
  data: MetricPoint[];
  width?: number;
  height?: number;
  selected?: boolean;
}

export function Chart({
  title,
  data,
  width = 40,
  height = 8,
  selected = false,
}: ChartProps): React.ReactElement {
  const borderColor = selected ? "cyan" : "gray";

  if (data.length === 0) {
    return (
      <Box
        flexDirection="column"
        width={width}
        borderStyle="round"
        borderColor={borderColor}
      >
        <Text bold color="cyan">
          {title}
        </Text>
        <Text dimColor>No data</Text>
      </Box>
    );
  }

  const values = data.map((p) => p.value);
  const lastStep = data[data.length - 1]?.step ?? 0;
  const lastValue = values[values.length - 1] ?? 0;

  const chartWidth = Math.max(width - 12, 10);
  const sampled = sampleData(values, chartWidth);

  const chart = asciichart.plot(sampled, {
    height: height - 4,
    format: (x: number) => x.toFixed(2).padStart(8),
  });

  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="round"
      borderColor={borderColor}
    >
      <Text bold color="cyan">
        {title}
      </Text>
      <Text>{chart}</Text>
      <Text dimColor>
        step: {lastStep} | value: {lastValue.toFixed(4)}
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
