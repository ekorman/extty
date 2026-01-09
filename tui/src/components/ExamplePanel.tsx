import React from "react";
import { Box, Text } from "ink";
import type { ExampleEntry } from "../lib/parser.js";

interface ExamplePanelProps {
  examples: Map<string, ExampleEntry[]>;
}

function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1)}…`;
}

function formatExample(entry: ExampleEntry): string[] {
  const prompt = entry.data.prompt;
  const response = entry.data.response;

  if (typeof prompt === "string" || typeof response === "string") {
    const lines = [];
    if (typeof prompt === "string") {
      lines.push(`prompt: ${truncate(prompt, 80)}`);
    }
    if (typeof response === "string") {
      lines.push(`response: ${truncate(response, 80)}`);
    }
    return lines;
  }

  return [truncate(JSON.stringify(entry.data), 120)];
}

export function ExamplePanel({ examples }: ExamplePanelProps): React.ReactElement | null {
  if (examples.size === 0) return null;

  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      <Text bold>Examples</Text>
      {Array.from(examples.entries()).map(([name, entries]) => {
        const latest = entries.slice(-3);
        return (
          <Box key={name} flexDirection="column" marginTop={1}>
            <Text color="cyan">{name}</Text>
            {latest.map((entry, index) => (
              <Box key={`${name}-${index}`} flexDirection="column" marginLeft={2}>
                <Text dimColor>
                  step {entry.step} • {new Date(entry.timestamp * 1000).toLocaleTimeString()}
                </Text>
                {formatExample(entry).map((line, lineIndex) => (
                  <Text key={`${name}-${index}-${lineIndex}`}>{line}</Text>
                ))}
              </Box>
            ))}
          </Box>
        );
      })}
    </Box>
  );
}
