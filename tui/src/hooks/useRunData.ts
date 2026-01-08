import { useState, useEffect } from "react";
import { getRunDir } from "../lib/paths.js";
import { loadRunData, type RunData } from "../lib/parser.js";
import { useFileWatch } from "./useFileWatch.js";

export function useRunData(runName: string | null): RunData | null {
  const [data, setData] = useState<RunData | null>(null);

  const runDir = runName ? getRunDir(runName) : "";
  const updateCount = useFileWatch(runDir, { enabled: !!runName });

  useEffect(() => {
    if (!runName) {
      setData(null);
      return;
    }

    const loaded = loadRunData(runName);
    setData(loaded);
  }, [runName, updateCount]);

  return data;
}
