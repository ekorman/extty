import { useState, useEffect, useRef } from "react";
import { getRunDir } from "../lib/paths.js";
import { loadRunData, type RunData } from "../lib/parser.js";
import { useFileWatch } from "./useFileWatch.js";

export function useRunData(runName: string | null): RunData | null {
  const [data, setData] = useState<RunData | null>(null);
  const [loadedRunName, setLoadedRunName] = useState<string | null>(null);

  const runDir = runName ? getRunDir(runName) : "";
  const updateCount = useFileWatch(runDir, {
    enabled: !!runName,
    debounceMs: 150,
  });

  useEffect(() => {
    if (!runName) {
      setData(null);
      setLoadedRunName(null);
      return;
    }

    const loaded = loadRunData(runName);
    setData(loaded);
    setLoadedRunName(runName);
  }, [runName, updateCount]);

  if (runName !== loadedRunName) {
    return null;
  }

  return data;
}
