import { homedir } from "node:os";
import { join } from "node:path";

export function getRunsDir(): string {
  return join(homedir(), ".extty", "runs");
}

export function getRunDir(runName: string): string {
  return join(getRunsDir(), runName);
}

export function getMetaPath(runName: string): string {
  return join(getRunDir(runName), "meta.json");
}

export function getMetricsDir(runName: string): string {
  return join(getRunDir(runName), "metrics");
}

export function getSystemPath(runName: string): string {
  return join(getRunDir(runName), "system.csv");
}

export function getExamplesDir(runName: string): string {
  return join(getRunDir(runName), "examples");
}
