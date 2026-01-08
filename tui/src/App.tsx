import React, { useState, useEffect } from "react";
import { Box, Text, useInput, useApp } from "ink";
import { useRunData } from "./hooks/useRunData.js";
import { listRuns } from "./lib/parser.js";
import { Chart } from "./components/Chart.js";
import { RunHeader } from "./components/RunHeader.js";
import { SystemStats } from "./components/SystemStats.js";
import { ConfigPanel } from "./components/ConfigPanel.js";
import { RunList } from "./components/RunList.js";
import { useFileWatch } from "./hooks/useFileWatch.js";
import { getRunsDir } from "./lib/paths.js";

type View = "list" | "detail";

interface AppProps {
  initialRun?: string;
}

export function App({ initialRun }: AppProps): React.ReactElement {
  const { exit } = useApp();
  const [runs, setRuns] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [view, setView] = useState<View>(initialRun ? "detail" : "list");

  useFileWatch(getRunsDir());

  useEffect(() => {
    const loadRuns = () => {
      const allRuns = listRuns();
      setRuns(allRuns);
      if (initialRun) {
        const idx = allRuns.indexOf(initialRun);
        if (idx >= 0) setSelectedIndex(idx);
      }
    };
    loadRuns();
    const interval = setInterval(loadRuns, 2000);
    return () => clearInterval(interval);
  }, [initialRun]);

  const currentRun = runs[selectedIndex] ?? null;
  const data = useRunData(view === "detail" ? currentRun : null);

  useInput((input, key) => {
    if (input === "q") {
      if (view === "detail") {
        setView("list");
      } else {
        exit();
      }
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
        setView("detail");
      }
    } else {
      if (key.leftArrow && selectedIndex > 0) {
        setSelectedIndex(selectedIndex - 1);
      }
      if (key.rightArrow && selectedIndex < runs.length - 1) {
        setSelectedIndex(selectedIndex + 1);
      }
    }
  });

  if (view === "list") {
    return <RunList runs={runs} selectedIndex={selectedIndex} />;
  }

  if (!data) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text>Loading run data...</Text>
      </Box>
    );
  }

  const metricNames = Array.from(data.metrics.keys());

  return (
    <Box flexDirection="column">
      <RunHeader meta={data.meta} metrics={data.metrics} />

      <Box flexDirection="row" flexWrap="wrap" gap={2} padding={1}>
        {metricNames.map((name) => (
          <Chart
            key={name}
            title={name}
            data={data.metrics.get(name) ?? []}
            width={45}
            height={10}
          />
        ))}
      </Box>

      <Box paddingTop={1}>
        <SystemStats system={data.system} />
      </Box>

      <ConfigPanel meta={data.meta} />

      <Box paddingX={1} paddingY={0} gap={2}>
        <Text dimColor>[q] back</Text>
        <Text dimColor>[←→] switch runs ({selectedIndex + 1}/{runs.length})</Text>
      </Box>
    </Box>
  );
}
