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

export interface ModelInfo {
  dir: string;
  name: string;
  n_embd: number;
  n_head: number;
  n_layer: number;
  n_ctx: number;
  vocab: number;
  params_m: number;
  size_mb: number;
  default: boolean;
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
  seqWarmupSteps: number;
  saveDir: string;
  outputDir: string;
  resume?: string;
}
