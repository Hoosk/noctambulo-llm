import type { JobState, LossPoint, TrainingEvent, TrainStatus } from "../api/types";

export interface LogEntry {
  id: number;
  text: string;
}

export interface JobInfo {
  status: JobState;
  paused: boolean;
  stopping: boolean;
  epoch: number;
  epochs: number;
  batch: number;
  numBatches: number;
  step: number;
  pct: number;
  loss: number | null;
  avgLoss: number | null;
  lr: number | null;
  batchesPerSec: number | null;
  elapsedSecs: number;
  etaSecs: number | null;
  nCtx: number;
  valPpl: number | null;
  valLoss: number | null;
  valPoints: LossPoint[];
  preview: string;
  jobId: string;
  message: string;
  lossPoints: LossPoint[];
  logs: LogEntry[];
  nextLogId: number;
}

export type Connection = "connecting" | "online" | "offline";
export type AuthState = "checking" | "gate" | "authed";

export const EMPTY_JOB: JobInfo = {
  status: "idle",
  paused: false,
  stopping: false,
  epoch: 0,
  epochs: 0,
  batch: 0,
  numBatches: 0,
  step: 0,
  pct: 0,
  loss: null,
  avgLoss: null,
  lr: null,
  batchesPerSec: null,
  elapsedSecs: 0,
  etaSecs: null,
  nCtx: 0,
  valPpl: null,
  valLoss: null,
  valPoints: [],
  preview: "",
  jobId: "",
  message: "",
  lossPoints: [],
  logs: [],
  nextLogId: 0,
};

// The trainer reports progress between loss-sync intervals. During that window
// its provisional loss is zero, which is not a real metric.
function metricOrNull(value: number): number | null {
  return Number.isFinite(value) && value > 0 ? value : null;
}

/// Append worker log lines (SSE or backfill) with stable ids for React keys.
export function appendLogLines(prev: JobInfo, lines: string[]): JobInfo {
  if (lines.length === 0) return prev;
  const entries = lines.map((text, i) => ({ id: prev.nextLogId + i, text }));
  const logs = [...prev.logs, ...entries];
  return {
    ...prev,
    logs: logs.length > 800 ? logs.slice(logs.length - 800) : logs,
    nextLogId: prev.nextLogId + entries.length,
  };
}

function pushLossPoint(points: LossPoint[], x: number, y: number | null): LossPoint[] {
  if (y == null) return points;
  const last = points[points.length - 1];
  if (last && last.x === x) {
    const next = points.slice(0, -1);
    next.push({ x, y });
    return next;
  }
  const next = points.length >= 2000 ? points.slice(points.length - 1999) : points.slice();
  next.push({ x, y });
  return next;
}

export function applyTrainingEvent(prev: JobInfo, ev: TrainingEvent): JobInfo {
  if ("Started" in ev) {
    const v = ev.Started;
    return {
      ...prev,
      status: "running",
      paused: false,
      stopping: false,
      epochs: v.epochs,
      epoch: 0,
      batch: 0,
      numBatches: v.num_batches,
      pct: 0,
      step: 0,
      loss: null,
      avgLoss: null,
      batchesPerSec: null,
      elapsedSecs: 0,
      etaSecs: null,
      valPpl: null,
      valLoss: null,
      preview: "",
      message: "Training started",
    };
  }
  if ("Progress" in ev) {
    const v = ev.Progress;
    const loss = metricOrNull(v.loss);
    return {
      ...prev,
      status: "running",
      epoch: v.epoch,
      epochs: v.epochs,
      batch: v.batch,
      numBatches: v.num_batches,
      step: v.step,
      pct: v.pct,
      loss,
      avgLoss: metricOrNull(v.avg_loss),
      lr: v.lr,
      batchesPerSec: v.batches_per_sec,
      elapsedSecs: v.elapsed_secs,
      etaSecs: v.eta_secs,
      nCtx: v.current_n_ctx,
      lossPoints: pushLossPoint(prev.lossPoints, v.step, loss),
    };
  }
  if ("Val" in ev) {
    const v = ev.Val;
    const loss = metricOrNull(v.loss);
    const x = prev.step > 0 ? prev.step : prev.epoch;
    return {
      ...prev,
      valPpl: v.ppl,
      valLoss: v.loss,
      valPoints: pushLossPoint(prev.valPoints, x, loss),
    };
  }
  if ("Preview" in ev) {
    return { ...prev, preview: ev.Preview.text };
  }
  if ("LogEvent" in ev) {
    return appendLogLines(prev, [ev.LogEvent.line]);
  }
  if ("Done" in ev) {
    return { ...prev, status: "done", stopping: false, message: "Training completed" };
  }
  if ("Stopped" in ev) {
    return { ...prev, status: "stopped", stopping: false, message: ev.Stopped.reason || "Stopped" };
  }
  if ("Checkpoint" in ev) {
    return { ...prev, message: `Checkpoint (${ev.Checkpoint.kind}) saved` };
  }
  return prev;
}

export function mergeStatus(prev: JobInfo, s: TrainStatus): { job: JobInfo; event: TrainingEvent | null } {
  let next = prev;
  if (s.paused !== prev.paused) {
    next = { ...next, paused: s.paused, message: s.paused ? "Training paused" : "Training resumed" };
  }
  if (Boolean(s.stopping) !== prev.stopping) {
    next = {
      ...next,
      stopping: Boolean(s.stopping),
      message: s.stopping ? "Stopping and saving checkpoint…" : next.message,
    };
  }
  if (Array.isArray(s.loss_points) && s.loss_points.length !== next.lossPoints.length) {
    next = { ...next, lossPoints: s.loss_points };
  }
  if (Array.isArray(s.val_points) && s.val_points.length !== next.valPoints.length) {
    next = { ...next, valPoints: s.val_points };
  }
  // A restarted server reports running=false while its persisted snapshot still
  // holds the last event of a (dead) job. Never let a stale Progress/Started
  // resurrect a ghost RUNNING state that would brick the UI.
  if (!s.running && next.status === "running") {
    next = { ...next, status: "idle", paused: false, message: "Job lost (server restarted?)" };
  }

  const ev = s.last_event;
  if (!ev) return { job: next, event: null };
  const staleLive = !s.running && ("Progress" in ev || "Started" in ev);
  if (staleLive) return { job: next, event: null };
  return { job: next, event: ev };
}
