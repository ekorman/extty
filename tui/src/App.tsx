import React, { useState, useEffect, useRef, useCallback } from "react";
import { Box, Text, useInput, useApp, useStdout } from "ink";
import { useRunData } from "./hooks/useRunData.js";
import { listRuns } from "./lib/parser.js";
import { Chart } from "./components/Chart.js";
import { RunHeader } from "./components/RunHeader.js";
import { SystemStats } from "./components/SystemStats.js";
import { ConfigPanel } from "./components/ConfigPanel.js";
import { RunList } from "./components/RunList.js";
import { ExamplePanel } from "./components/ExamplePanel.js";
import { useFileWatch } from "./hooks/useFileWatch.js";
import { getRunsDir } from "./lib/paths.js";
import { pushRun } from "./lib/sync.js";

type View = "list" | "detail";

interface AppProps {
  initialRun?: string;
}

export function App({ initialRun }: AppProps): React.ReactElement {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [view, setView] = useState<View>(initialRun ? "detail" : "list");
  const prevDataRef = useRef<typeof data>(null);
  const [pushStatus, setPushStatus] = useState<{
    message: string;
    success: boolean;
  } | null>(null);
  const [selectedChartIndex, setSelectedChartIndex] = useState(0);
  const [chartFocused, setChartFocused] = useState(false);

  // Clear screen helper to prevent leftover content when switching views
  const clearScreen = useCallback(() => {
    stdout.write("\x1b[2J\x1b[3J\x1b[H");
  }, [stdout]);

  const runsUpdateCount = useFileWatch(getRunsDir(), { debounceMs: 200 });

  const loadRuns = useCallback(() => {
    const allRuns = listRuns();
    setRuns((prev) => {
      // Only update if the list actually changed to prevent unnecessary re-renders
      if (
        prev.length === allRuns.length &&
        prev.every((r, i) => r === allRuns[i])
      ) {
        return prev;
      }
      return allRuns;
    });
    if (initialRun) {
      const idx = allRuns.indexOf(initialRun);
      if (idx >= 0) setSelectedIndex(idx);
    }
  }, [initialRun]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns, runsUpdateCount]);

  const currentRun = runs[selectedIndex] ?? null;
  const data = useRunData(view === "detail" ? currentRun : null);

  // Keep reference to previous data to avoid flicker during updates
  if (data) {
    prevDataRef.current = data;
  }
  const displayData = data ?? prevDataRef.current;
  const handlePush = async (runName: string) => {
    setPushStatus({ message: "Pushing...", success: true });
    const result = await pushRun(runName);
    setPushStatus({ message: result.message, success: result.success });
    setTimeout(() => setPushStatus(null), 3000);
  };

  const metricNames = displayData ? Array.from(displayData.metrics.keys()) : [];
  const chartCount = metricNames.length;

  useInput((input, key) => {
    if (input === "q" || key.escape) {
      if (chartFocused) {
        clearScreen();
        setChartFocused(false);
      } else if (view === "detail") {
        clearScreen();
        setView("list");
        prevDataRef.current = null;
        setSelectedChartIndex(0);
      } else {
        exit();
      }
      return;
    }

    if (input === "p" && currentRun) {
      handlePush(currentRun);
      return;
    }

    if (view === "list") {
      if (key.upArrow && selectedIndex > 0) {
        setSelectedIndex(selectedIndex - 1);
      }
      if (key.downArrow && selectedIndex < runs.length - 1) {
        setSelectedIndex(selectedIndex + 1);
      }
      if (key.return && runs.length > 0) {
        clearScreen();
        setView("detail");
      }
    } else if (chartFocused) {
      if (key.leftArrow && selectedChartIndex > 0) {
        setSelectedChartIndex(selectedChartIndex - 1);
      }
      if (key.rightArrow && selectedChartIndex < chartCount - 1) {
        setSelectedChartIndex(selectedChartIndex + 1);
      }
    } else {
      if (key.leftArrow && selectedChartIndex > 0) {
        setSelectedChartIndex(selectedChartIndex - 1);
      }
      if (key.rightArrow && selectedChartIndex < chartCount - 1) {
        setSelectedChartIndex(selectedChartIndex + 1);
      }
      if (key.upArrow && selectedChartIndex > 0) {
        setSelectedChartIndex(Math.max(0, selectedChartIndex - 1));
      }
      if (key.downArrow && selectedChartIndex < chartCount - 1) {
        setSelectedChartIndex(Math.min(chartCount - 1, selectedChartIndex + 1));
      }
      if (key.return && chartCount > 0) {
        clearScreen();
        setChartFocused(true);
      }
      if (input === "[" && selectedIndex > 0) {
        clearScreen();
        prevDataRef.current = null;
        setSelectedChartIndex(0);
        setSelectedIndex(selectedIndex - 1);
      }
      if (input === "]" && selectedIndex < runs.length - 1) {
        clearScreen();
        prevDataRef.current = null;
        setSelectedChartIndex(0);
        setSelectedIndex(selectedIndex + 1);
      }
    }
  });

  if (view === "list") {
    return (
      <RunList
        runs={runs}
        selectedIndex={selectedIndex}
        pushStatus={pushStatus}
      />
    );
  }

  if (!displayData) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text>Loading run data...</Text>
      </Box>
    );
  }

  const termWidth = stdout.columns ?? 80;
  const termHeight = stdout.rows ?? 24;

  if (chartFocused) {
    const focusedName = metricNames[selectedChartIndex];
    const focusedData = displayData.metrics.get(focusedName) ?? [];
    return (
      <Box flexDirection="column" padding={1}>
        <Chart
          title={focusedName}
          data={focusedData}
          width={termWidth - 4}
          height={termHeight - 6}
        />
        <Box paddingTop={1} gap={2}>
          <Text dimColor>[q] back to grid</Text>
          <Text dimColor>
            [←→] chart {selectedChartIndex + 1}/{chartCount}
          </Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      <RunHeader meta={displayData.meta} metrics={displayData.metrics} />

      <Box flexDirection="row" flexWrap="wrap" gap={2} padding={1}>
        {metricNames.map((name, index) => (
          <Chart
            key={name}
            title={name}
            data={displayData.metrics.get(name) ?? []}
            width={45}
            height={10}
            selected={index === selectedChartIndex}
          />
        ))}
      </Box>

      <Box paddingTop={1}>
        <SystemStats system={displayData.system} />
      </Box>

      <ExamplePanel examples={displayData.examples} />

      <ConfigPanel meta={displayData.meta} />

      {pushStatus && (
        <Box paddingX={1}>
          <Text color={pushStatus.success ? "green" : "red"}>
            {pushStatus.message}
          </Text>
        </Box>
      )}

      <Box paddingX={1} paddingY={0} gap={2}>
        <Text dimColor>[q] back</Text>
        <Text dimColor>[←→↑↓] select chart</Text>
        <Text dimColor>[Enter] focus</Text>
        <Text dimColor>
          [&#91;&#93;] run {selectedIndex + 1}/{runs.length}
        </Text>
        <Text dimColor>[p] push</Text>
      </Box>
    </Box>
  );
}
