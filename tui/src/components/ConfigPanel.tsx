import React from "react";
import { Box, Text } from "ink";
import type { RunMeta } from "../lib/parser.js";

interface ConfigPanelProps {
  meta: RunMeta | null;
}

export function ConfigPanel({ meta }: ConfigPanelProps): React.ReactElement {
  if (!meta || Object.keys(meta.config).length === 0) {
    return <></>;
  }

  const configStr = Object.entries(meta.config)
    .map(([k, v]) => `${k}=${formatValue(v)}`)
    .join(", ");

  return (
    <Box paddingX={1}>
      <Text dimColor>Config: </Text>
      <Text>{configStr}</Text>
    </Box>
  );
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toString() : v.toFixed(4);
  }
  if (typeof v === "boolean") {
    return v ? "true" : "false";
  }
  if (typeof v === "string") {
    return v;
  }
  return JSON.stringify(v);
}
