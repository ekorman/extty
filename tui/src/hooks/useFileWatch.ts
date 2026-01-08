import { useState, useEffect, useCallback } from "react";
import chokidar from "chokidar";

export function useFileWatch(
  path: string,
  options: { enabled?: boolean } = {}
): number {
  const { enabled = true } = options;
  const [updateCount, setUpdateCount] = useState(0);

  const triggerUpdate = useCallback(() => {
    setUpdateCount((c) => c + 1);
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const watcher = chokidar.watch(path, {
      persistent: true,
      ignoreInitial: true,
      depth: 2,
    });

    watcher.on("add", triggerUpdate);
    watcher.on("change", triggerUpdate);
    watcher.on("unlink", triggerUpdate);

    return () => {
      watcher.close();
    };
  }, [path, enabled, triggerUpdate]);

  return updateCount;
}
