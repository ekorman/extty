import React from "react";
import { Box, Text } from "ink";
import { loadRunData } from "../lib/parser.js";

interface RunListProps {
  runs: string[];
  selectedIndex: number;
  pushStatus?: { message: string; success: boolean } | null;
}

export function RunList({ runs, selectedIndex, pushStatus }: RunListProps): React.ReactElement {
  if (runs.length === 0) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="yellow">extty</Text>
        <Text />
        <Text dimColor>No runs found in ~/.extty/runs/</Text>
        <Text dimColor>Start a training run with the extty Python library.</Text>
        <Text />
        <Text dimColor>Press q to quit</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      <Text bold color="yellow">extty</Text>
      <Text dimColor>Select a run to view:</Text>
      <Text />

      {runs.map((run, i) => (
        <RunListItem
          key={run}
          name={run}
          selected={i === selectedIndex}
        />
      ))}

      {pushStatus && (
        <Box>
          <Text color={pushStatus.success ? "green" : "red"}>{pushStatus.message}</Text>
        </Box>
      )}

      <Text />
      <Box gap={2}>
        <Text dimColor>[↑↓] navigate</Text>
        <Text dimColor>[enter] select</Text>
        <Text dimColor>[p] push</Text>
        <Text dimColor>[q] quit</Text>
      </Box>
    </Box>
  );
}

interface RunListItemProps {
  name: string;
  selected: boolean;
}

function RunListItem({ name, selected }: RunListItemProps): React.ReactElement {
  const data = loadRunData(name);
  const status = data.meta?.status ?? "unknown";
  const project = data.meta?.project ?? "?";

  const statusColor = status === "running"
    ? "green"
    : status === "completed"
      ? "blue"
      : "yellow";

  const statusIcon = status === "running" ? "●" : status === "completed" ? "✓" : "?";

  return (
    <Box>
      <Text color={selected ? "cyan" : undefined}>
        {selected ? "❯ " : "  "}
      </Text>
      <Text color={statusColor}>{statusIcon} </Text>
      <Text bold={selected}>{name}</Text>
      <Text dimColor> ({project})</Text>
      <Text dimColor> - </Text>
      <Text color={statusColor}>{status}</Text>
    </Box>
  );
}
