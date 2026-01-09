import { useState, useEffect, useRef } from "react";
import { getRunDir } from "../lib/paths.js";
import { loadRunData, type RunData } from "../lib/parser.js";
import { useFileWatch } from "./useFileWatch.js";

export function useRunData(runName: string | null): RunData | null {
  const [data, setData] = useState<RunData | null>(null);
  const prevRunName = useRef<string | null>(null);

  const runDir = runName ? getRunDir(runName) : "";
  const updateCount = useFileWatch(runDir, {
    enabled: !!runName,
    debounceMs: 150,
  });

  useEffect(() => {
    if (!runName) {
      setData(null);
      prevRunName.current = null;
      return;
    }

    // Only clear data when switching to a different run
    // This prevents flicker during updates to the same run
    const isNewRun = prevRunName.current !== runName;
    prevRunName.current = runName;

    const loaded = loadRunData(runName);

    // Use functional update to batch with any pending updates
    setData((prev) => {
      // If loading failed and we have previous data for same run, keep it
      if (!loaded && prev && !isNewRun) {
        return prev;
      }
      return loaded;
    });
  }, [runName, updateCount]);

  return data;
}
