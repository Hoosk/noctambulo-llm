import type {
  CheckpointInfo,
  CombineRequest,
  DatasetDetail,
  DatasetHit,
  DatasetInfo,
  DeviceInfo,
  ExportRequestBody,
  LogsResponse,
  ModelFilesResponse,
  ModelInfo,
  OpStatus,
  RunRecord,
  RunSummary,
  Telemetry,
  TrainRequestBody,
  TrainStatus,
} from "./types";

export const UNAUTHORIZED_EVENT = "kuro:unauthorized";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface RequestOptions {
  method?: string;
  json?: unknown;
  signal?: AbortSignal;
}

// Auth travels in the HttpOnly session cookie set by /api/login: fetch sends it
// automatically on same-origin requests and JavaScript can never read it.
export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers = new Headers();
  let body: string | undefined;
  if (opts.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(opts.json);
  }

  const res = await fetch(path, {
    method: opts.method ?? "GET",
    headers,
    body,
    signal: opts.signal,
  });

  if (res.status === 401) {
    // Session revoked/expired mid-session: let the app show the gate.
    window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
    throw new ApiError(401, "Invalid or missing token");
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const parsed = (await res.json()) as { error?: string };
      if (parsed.error) detail = parsed.error;
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return null as T;
  return (await res.json()) as T;
}

export const api = {
  login: (token: string) => request<{ ok: boolean }>("/api/login", { method: "POST", json: { token } }),
  logout: () => request<{ ok: boolean }>("/api/logout", { method: "POST" }),
  ping: () => request<Record<string, unknown>>("/api/ping"),
  getStatus: () => request<TrainStatus>("/api/train/status"),
  getLogs: (since = 0) => request<LogsResponse>(`/api/train/logs?since=${since}`),
  getDatasets: () => request<DatasetInfo[]>("/api/datasets"),
  getDatasetInfo: (name: string) =>
    request<DatasetDetail>(`/api/datasets/info/${encodeURIComponent(name)}`),
  searchDatasets: (q: string) =>
    request<DatasetHit[]>(`/api/datasets/search?q=${encodeURIComponent(q)}`),
  downloadDataset: (repoId: string, filename: string) =>
    request<{ opId: string; status: string }>("/api/datasets/download", {
      method: "POST",
      json: { repoId, filename },
    }),
  renameDataset: (from: string, to: string) =>
    request<{ status: string }>("/api/datasets/rename", { method: "POST", json: { from, to } }),
  deleteDataset: (name: string) =>
    request<{ status: string; deleted: string }>(`/api/datasets/delete/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  combineDatasets: (body: CombineRequest) =>
    request<{ opId: string; status: string }>("/api/datasets/combine", { method: "POST", json: body }),
  getDevices: () => request<DeviceInfo[]>("/api/devices"),
  getTelemetry: () => request<Telemetry>("/api/telemetry"),
  getModels: () => request<ModelInfo[]>("/api/models"),
  getModelFiles: (dir: string) =>
    request<ModelFilesResponse>(`/api/models/files?dir=${encodeURIComponent(dir)}`),
  modelFileUrl: (dir: string, file: string) =>
    `/api/models/file?dir=${encodeURIComponent(dir)}&file=${encodeURIComponent(file)}`,
  renameModel: (dir: string, newName: string) =>
    request<{ status: string; dir: string; name?: string }>("/api/models/rename", {
      method: "POST",
      json: { dir, newName },
    }),
  deleteModel: (name: string) =>
    request<{ status: string; deleted: string }>(`/api/models/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  exportModel: (body: ExportRequestBody) =>
    request<{ opId: string; status: string }>("/api/export", { method: "POST", json: body }),
  getOps: () => request<OpStatus[]>("/api/ops"),
  cancelOp: (id: string) =>
    request<{ status: string; id: string }>(`/api/ops/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  getRuns: () => request<RunSummary[]>("/api/runs"),
  getRun: (id: string) => request<RunRecord>(`/api/runs/${encodeURIComponent(id)}`),
  getCheckpoints: (dir = "checkpoints_web") =>
    request<CheckpointInfo[]>(`/api/checkpoints?dir=${encodeURIComponent(dir)}`),
  startTrain: (body: TrainRequestBody) =>
    request<{ jobId: string; status: string }>("/api/train", { method: "POST", json: body }),
  stopTrain: () => request<{ status: string }>("/api/train/stop", { method: "POST" }),
  pauseTrain: () => request<{ status: string }>("/api/train/pause", { method: "POST" }),
  resumeTrain: () => request<{ status: string }>("/api/train/resume", { method: "POST" }),
};
