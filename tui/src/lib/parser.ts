import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { getMetaPath, getMetricsDir, getSystemPath, getRunsDir } from "./paths.js";

export interface MetricPoint {
  step: number;
  timestamp: number;
  value: number;
}

export interface SystemPoint {
  timestamp: number;
  ramUsedGb: number;
  ramTotalGb: number;
  gpuMemUsedGb: number | null;
  gpuMemTotalGb: number | null;
  gpuUtilPct: number | null;
}

export interface RunMeta {
  project: string;
  runName: string;
  config: Record<string, unknown>;
  startedAt: string;
  finishedAt: string | null;
  status: "running" | "completed" | "failed";
}

export interface RunData {
  meta: RunMeta | null;
  metrics: Map<string, MetricPoint[]>;
  system: SystemPoint[];
}

export function parseMetricsCsv(content: string): MetricPoint[] {
  const lines = content.trim().split("\n");
  if (lines.length <= 1) return [];

  const points: MetricPoint[] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length >= 3) {
      points.push({
        step: parseInt(parts[0], 10),
        timestamp: parseFloat(parts[1]),
        value: parseFloat(parts[2]),
      });
    }
  }
  return points;
}

export function parseSystemCsv(content: string): SystemPoint[] {
  const lines = content.trim().split("\n");
  if (lines.length <= 1) return [];

  const points: SystemPoint[] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length >= 3) {
      points.push({
        timestamp: parseFloat(parts[0]),
        ramUsedGb: parseFloat(parts[1]),
        ramTotalGb: parseFloat(parts[2]),
        gpuMemUsedGb: parts[3] ? parseFloat(parts[3]) : null,
        gpuMemTotalGb: parts[4] ? parseFloat(parts[4]) : null,
        gpuUtilPct: parts[5] ? parseFloat(parts[5]) : null,
      });
    }
  }
  return points;
}

export function parseMetaJson(content: string): RunMeta | null {
  try {
    const data = JSON.parse(content);
    return {
      project: data.project,
      runName: data.run_name,
      config: data.config || {},
      startedAt: data.started_at,
      finishedAt: data.finished_at,
      status: data.status,
    };
  } catch {
    return null;
  }
}

function findCsvFiles(dir: string, baseDir: string): string[] {
  const results: string[] = [];
  if (!existsSync(dir)) return results;

  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findCsvFiles(fullPath, baseDir));
    } else if (entry.isFile() && entry.name.endsWith(".csv")) {
      results.push(relative(baseDir, fullPath));
    }
  }
  return results;
}

export function loadRunData(runName: string): RunData {
  const metaPath = getMetaPath(runName);
  const metricsDir = getMetricsDir(runName);
  const systemPath = getSystemPath(runName);

  let meta: RunMeta | null = null;
  if (existsSync(metaPath)) {
    meta = parseMetaJson(readFileSync(metaPath, "utf-8"));
  }

  const metrics = new Map<string, MetricPoint[]>();
  const csvFiles = findCsvFiles(metricsDir, metricsDir);
  for (const relativePath of csvFiles) {
    const metricName = relativePath.replace(/\.csv$/, "");
    const content = readFileSync(join(metricsDir, relativePath), "utf-8");
    metrics.set(metricName, parseMetricsCsv(content));
  }

  let system: SystemPoint[] = [];
  if (existsSync(systemPath)) {
    system = parseSystemCsv(readFileSync(systemPath, "utf-8"));
  }

  return { meta, metrics, system };
}

export function listRuns(): string[] {
  const runsDir = getRunsDir();
  if (!existsSync(runsDir)) return [];

  return readdirSync(runsDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name)
    .sort()
    .reverse();
}
