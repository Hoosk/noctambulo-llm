export interface LossPoint {
  x: number;
  y: number;
}

export interface LogLine {
  line: string;
  ts: number;
}

export type TrainingEvent =
  | {
      Started: {
        epochs: number;
        total_steps: number;
        num_batches: number;
        batch_size: number;
        lr: number;
        n_ctx: number;
        start_time_unix: number;
      };
    }
  | {
      Progress: {
        step: number;
        epoch: number;
        epochs: number;
        batch: number;
        num_batches: number;
        pct: number;
        loss: number;
        avg_loss: number;
        lr: number;
        batches_per_sec: number;
        elapsed_secs: number;
        eta_secs: number;
        current_n_ctx: number;
      };
    }
  | { Val: { loss: number; ppl: number; bpb: number | null; epoch: number } }
  | { Checkpoint: { path: string; kind: string; epoch: number; batch: number } }
  | { Preview: { text: string; epoch: number } }
  | { EpochComplete: { epoch: number; epochs: number; avg_loss: number; elapsed_secs: number } }
  | {
      Done: {
        final_loss: number;
        total_time_secs: number;
        weights_path: string;
        config_path: string;
      };
    }
  | { Stopped: { reason: string; epoch: number; batch: number } }
  | { LogEvent: { line: string; ts: number } };

export interface TrainStatus {
  running: boolean;
  stopping: boolean;
  paused: boolean;
  job_id: string;
  message: string | null;
  last_event: TrainingEvent | null;
  loss_points: LossPoint[];
  val_points: LossPoint[];
  logs_total: number;
}

export interface LogsResponse {
  lines: LogLine[];
  total: number;
}

export type JobState = "idle" | "running" | "done" | "stopped" | "error";

export interface DatasetInfo {
  name: string;
  size_bytes: number;
  modified_unix: number;
  ext: string;
}

export interface DatasetHit {
  id: string;
  downloads: number;
  likes: number;
  last_modified: string;
  tags: string[];
}

export interface RepoFiles {
  files: string[];
  default: string | null;
}

export interface DatasetDetail {
  name: string;
  size_bytes: number;
  ext: string;
  lines?: number | null;
}

export interface CombineSource {
  path: string;
  weight: number;
}

export interface CombineRequest {
  sources: CombineSource[];
  output: string;
  maxChars?: number;
}

export interface DeviceInfo {
  name: string;
  backend: string;
  device_type: string;
  vendor: number;
  device: number;
  driver: string;
  vram_bytes: number | null;
  selectable: boolean;
  selector: string;
}

export interface CheckpointInfo {
  name: string;
  path: string;
  size_bytes: number;
  modified_unix: number;
}

export interface DiskInfo {
  name: string;
  mount_point: string;
  total_bytes: number;
  available_bytes: number;
}

export interface AcceleratorInfo {
  name: string;
  driver: string;
  pci_id: string;
  slot: string;
}

export interface SystemInfo {
  hostname: string;
  os_name: string;
  os_version: string;
  kernel_version: string;
  cpu_brand: string;
  cpu_physical_cores: number | null;
  cpu_logical_cores: number;
  cpu_frequency_mhz: number;
  cpu_usage_pct: number;
  cpu_per_core: number[];
  total_memory_bytes: number;
  used_memory_bytes: number;
  total_swap_bytes: number;
  used_swap_bytes: number;
  uptime_secs: number;
  load_average: [number, number, number];
  disks: DiskInfo[];
  accelerators: AcceleratorInfo[];
}

export interface ModelInfo {
  dir: string;
  name: string;
  n_embd: number;
  n_head: number;
  n_layer: number;
  n_ctx: number;
  vocab: number;
  ffn_hidden: number;
  rope_base: number;
  rms_norm_eps: number;
  import_source: string | null;
  params_m: number;
  size_mb: number;
  default: boolean;
  arch_version: number;
  compatible: boolean;
  legacy_weights: boolean;
  steps: number | null;
  best_val_loss: number | null;
}

export interface ModelFile {
  name: string;
  size_bytes: number;
  modified_unix: number;
  shared: boolean;
}

export interface ModelFilesResponse {
  dir: string;
  name: string;
  files: ModelFile[];
}

export interface TrainRequestBody {
  file: string;
  device: string;
  dtype: string;
  earlyStopPatience: number;
  epochs: number;
  batchSize: number;
  lr: number;
  maxTokens: number | null;
  nCtx: number;
  nEmbd: number;
  nHead: number;
  nLayer: number;
  valSplit: number;
  gradAccum: number;
  nKvHead: number | null;
  dropout: number;
  attnSoftcap: number;
  slidingWindow?: number;
  sinkTokens?: number;
  kvShareGroup?: number;
  mlaLatentDim?: number;
  kvEviction?: string;
  kvBudget?: number;
  kvRecent?: number;
  seqWarmupSteps: number;
  weightDecay: number;
  labelSmoothing: number;
  zLoss: number;
  saveDir: string;
  outputDir: string;
  resume?: string;
}

export interface ExportRequestBody {
  modelDir: string;
  modelName: string;
  approx?: boolean;
}

export type OpEvent =
  | { Started: { id: string; kind: string; label: string; started_unix: number } }
  | { Progress: { id: string; kind: string; phase: string; done: number; total: number; unit: string } }
  | { Done: { id: string; kind: string; message: string; result: Record<string, unknown> | null } }
  | { Failed: { id: string; kind: string; error: string } }
  | { Cancelled: { id: string; kind: string } };

export interface OpStatus {
  id: string;
  kind: string;
  label: string;
  phase: string;
  done: number;
  total: number;
  unit: string;
  status: "running" | "done" | "failed" | "cancelled";
  message: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
  started_unix: number;
  finished_unix: number | null;
}

export interface RunSummary {
  id: string;
  job_id: string;
  dataset: string;
  output_dir: string;
  dtype: string;
  device: string;
  epochs: number;
  batch_size: number;
  lr: number;
  n_ctx: number;
  n_embd: number;
  n_layer: number;
  grad_accum: number;
  status: string;
  note: string | null;
  started_unix: number;
  finished_unix: number | null;
  final_loss: number | null;
  final_ppl: number | null;
}

export interface RunRecord extends RunSummary {
  loss_points: LossPoint[];
  val_points: LossPoint[];
}

export interface TelemetryMemory {
  total_bytes: number;
  available_bytes: number;
  used_pct: number;
}

export interface TelemetryProcess {
  rss_bytes: number;
  cpu_pct: number | null;
  gpu_pct: number | null;
  threads: number;
  uptime_secs: number;
}

export interface TelemetryGpu {
  index: number;
  name: string;
  used_bytes: number;
  total_bytes: number;
  utilization_pct: number | null;
  source: string;
}

export interface Telemetry {
  memory: TelemetryMemory;
  process: TelemetryProcess;
  gpus: TelemetryGpu[];
  notes: string[];
}
