const API_BASE = "/api/v1";

export interface Run {
  id: number;
  project_id: number;
  project_name: string | null;
  name: string;
  config: Record<string, unknown> | null;
  started_at: string | null;
  finished_at: string | null;
  status: string;
  created_at: string;
}

export interface RunDetail extends Run {
  metrics: string[];
  system_points: SystemPoint[];
}

export interface MetricPoint {
  step: number;
  value: number;
  timestamp: number;
}

export interface SystemPoint {
  timestamp: number;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  gpu_mem_used_gb: number | null;
  gpu_mem_total_gb: number | null;
  gpu_util_pct: number | null;
}

export interface Project {
  id: number;
  name: string;
  created_at: string;
}

export interface RunFilters {
  project?: string;
  status?: string;
  search?: string;
  sort?: string;
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchRuns(filters: RunFilters = {}): Promise<Run[]> {
  const params = new URLSearchParams();
  if (filters.project) params.set("project", filters.project);
  if (filters.status) params.set("status", filters.status);
  if (filters.search) params.set("search", filters.search);
  if (filters.sort) params.set("sort", filters.sort);
  if (filters.order) params.set("order", filters.order);
  if (filters.limit) params.set("limit", filters.limit.toString());
  if (filters.offset) params.set("offset", filters.offset.toString());

  const query = params.toString();
  const url = `${API_BASE}/runs${query ? `?${query}` : ""}`;
  return fetchJson<Run[]>(url);
}

export async function fetchRun(id: number): Promise<RunDetail> {
  return fetchJson<RunDetail>(`${API_BASE}/runs/${id}`);
}

export async function fetchMetricPoints(runId: number, metricName: string): Promise<MetricPoint[]> {
  return fetchJson<MetricPoint[]>(`${API_BASE}/runs/${runId}/metrics/${encodeURIComponent(metricName)}`);
}

export async function fetchProjects(): Promise<Project[]> {
  return fetchJson<Project[]>(`${API_BASE}/projects`);
}

export async function deleteRun(id: number): Promise<void> {
  await fetch(`${API_BASE}/runs/${id}`, { method: "DELETE" });
}
