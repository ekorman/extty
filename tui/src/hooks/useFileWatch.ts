import { useState, useEffect, useCallback, useRef } from "react";
import chokidar from "chokidar";

export function useFileWatch(
  path: string,
  options: { enabled?: boolean; debounceMs?: number } = {}
): number {
  const { enabled = true, debounceMs = 100 } = options;
  const [updateCount, setUpdateCount] = useState(0);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerUpdate = useCallback(() => {
    // Debounce updates to prevent rapid re-renders causing flicker
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      setUpdateCount((c) => c + 1);
      debounceRef.current = null;
    }, debounceMs);
  }, [debounceMs]);

  useEffect(() => {
    if (!enabled) return;

    const watcher = chokidar.watch(path, {
      persistent: true,
      ignoreInitial: true,
      depth: 2,
      // Use polling with a reasonable interval to reduce file system events
      usePolling: false,
      // Stabilize events - wait for file writes to complete
      awaitWriteFinish: {
        stabilityThreshold: 100,
        pollInterval: 50,
      },
    });

    watcher.on("add", triggerUpdate);
    watcher.on("change", triggerUpdate);
    watcher.on("unlink", triggerUpdate);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      watcher.close();
    };
  }, [path, enabled, triggerUpdate]);

  return updateCount;
}
