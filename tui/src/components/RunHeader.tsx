import React from "react";
import { Box, Text } from "ink";
import type { RunMeta, MetricPoint } from "../lib/parser.js";

interface RunHeaderProps {
  meta: RunMeta | null;
  metrics: Map<string, MetricPoint[]>;
}

export function RunHeader({
  meta,
  metrics,
}: RunHeaderProps): React.ReactElement {
  const runName = meta?.runName ?? "Unknown";
  const status = meta?.status ?? "unknown";

  const statusColor =
    status === "running" ? "green" : status === "completed" ? "blue" : "yellow";

  let maxStep = 0;
  for (const points of metrics.values()) {
    for (const p of points) {
      if (p.step > maxStep) maxStep = p.step;
    }
  }

  return (
    <Box borderStyle="single" paddingX={1}>
      <Text bold>extty</Text>
      <Text> - </Text>
      <Text color="cyan">{runName}</Text>
      <Text> (</Text>
      <Text color={statusColor}>{status}</Text>
      <Text>)</Text>
      <Box flexGrow={1} />
      <Text>step: </Text>
      <Text bold color="yellow">
        {maxStep.toLocaleString()}
      </Text>
    </Box>
  );
}
