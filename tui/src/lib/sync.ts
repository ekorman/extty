import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { getRunDir, getMetaPath, getMetricsDir, getSystemPath } from "./paths.js";
import { parseMetricsCsv, parseSystemCsv } from "./parser.js";

interface PushResult {
  success: boolean;
  message: string;
}

function findCsvFilesRecursive(dir: string, baseDir: string, results: string[] = []): string[] {
  if (!existsSync(dir)) return results;

  const { readdirSync } = require("node:fs");
  const { relative } = require("node:path");

  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      findCsvFilesRecursive(fullPath, baseDir, results);
    } else if (entry.isFile() && entry.name.endsWith(".csv")) {
      results.push(relative(baseDir, fullPath));
    }
  }
  return results;
}

function buildPayload(runName: string): Record<string, unknown> {
  const metaPath = getMetaPath(runName);
  const metricsDir = getMetricsDir(runName);
  const systemPath = getSystemPath(runName);

  if (!existsSync(metaPath)) {
    throw new Error(`meta.json not found for run '${runName}'`);
  }

  const meta = JSON.parse(readFileSync(metaPath, "utf-8"));

  const payload: Record<string, unknown> = {
    project: meta.project ?? "unknown",
    run_name: meta.run_name ?? runName,
    config: meta.config,
    started_at: meta.started_at,
    finished_at: meta.finished_at,
    status: meta.status ?? "completed",
    metrics: {} as Record<string, unknown[]>,
    system: [] as unknown[],
  };

  const csvFiles = findCsvFilesRecursive(metricsDir, metricsDir);
  for (const relativePath of csvFiles) {
    const metricName = relativePath.replace(/\.csv$/, "");
    const content = readFileSync(join(metricsDir, relativePath), "utf-8");
    const points = parseMetricsCsv(content);
    if (points.length > 0) {
      (payload.metrics as Record<string, unknown[]>)[metricName] = points;
    }
  }

  if (existsSync(systemPath)) {
    const systemContent = readFileSync(systemPath, "utf-8");
    payload.system = parseSystemCsv(systemContent);
  }

  return payload;
}

export async function pushRun(
  runName: string,
  serverUrl?: string,
  apiKey?: string
): Promise<PushResult> {
  const server = serverUrl ?? process.env.EXTTY_SERVER;
  if (!server) {
    return {
      success: false,
      message: "Server URL required. Set EXTTY_SERVER env var.",
    };
  }

  const key = apiKey ?? process.env.EXTTY_API_KEY;
  const runDir = getRunDir(runName);

  if (!existsSync(runDir)) {
    return {
      success: false,
      message: `Run '${runName}' not found`,
    };
  }

  let payload: Record<string, unknown>;
  try {
    payload = buildPayload(runName);
  } catch (err) {
    return {
      success: false,
      message: `Failed to build payload: ${err}`,
    };
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (key) {
    headers["Authorization"] = `Bearer ${key}`;
  }

  try {
    const response = await fetch(`${server.replace(/\/$/, "")}/api/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      return {
        success: true,
        message: `Pushed '${runName}' to ${server}`,
      };
    } else {
      const body = await response.text();
      return {
        success: false,
        message: `Push failed (${response.status}): ${body}`,
      };
    }
  } catch (err) {
    return {
      success: false,
      message: `Connection failed: ${err}`,
    };
  }
}
