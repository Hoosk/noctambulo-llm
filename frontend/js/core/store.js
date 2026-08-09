// Global store for the training job state (fed by SSE events).

import { emit, on } from "./events.js";

export const JobState = {
  IDLE: "idle",
  RUNNING: "running",
  DONE: "done",
  STOPPED: "stopped",
  ERROR: "error",
};

const store = {
  status: JobState.IDLE,
  paused: false,
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
  preview: "",
  jobId: "",
  message: "",
  lossPoints: [],
};

export function getJob() {
  return store;
}

function set(patch) {
  Object.assign(store, patch);
  emit("job:changed", store);
}

// Translate a raw TrainingEvent into store mutations.
export function handleTrainingEvent(ev) {
  const key = Object.keys(ev)[0];
  const v = ev[key];
  switch (key) {
    case "Started":
      set({
        status: JobState.RUNNING,
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
        message: "Entrenamiento iniciado",
      });
      break;
    case "Progress":
      set({
        status: JobState.RUNNING,
        epoch: v.epoch,
        epochs: v.epochs,
        batch: v.batch,
        numBatches: v.num_batches,
        step: v.step,
        pct: v.pct,
        loss: metricOrNull(v.loss),
        avgLoss: metricOrNull(v.avg_loss),
        lr: v.lr,
        batchesPerSec: v.batches_per_sec,
        elapsedSecs: v.elapsed_secs,
        etaSecs: v.eta_secs,
        nCtx: v.current_n_ctx,
      });
      break;
    case "Val":
      set({ valPpl: v.ppl, valLoss: v.loss });
      break;
    case "Preview":
      set({ preview: v.text });
      break;
    case "Done":
      set({ status: JobState.DONE, message: "Entrenamiento completado" });
      break;
    case "Stopped":
      set({ status: JobState.STOPPED, message: v.reason || "Detenido" });
      break;
    case "Checkpoint":
      set({ message: `Checkpoint ${v.kind} guardado` });
      break;
    default:
      break;
  }
}

// The trainer reports progress between loss-sync intervals. During that
// window its provisional loss is zero, which is not a real metric.
function metricOrNull(value) {
  return Number.isFinite(value) && value > 0 ? value : null;
}

// Wire SSE events into the store once.
export function initJobStore() {
  on("training:event", handleTrainingEvent);
}

// ---- Polling fallback -----------------------------------------------------
// The SSE stream is the primary channel, but browsers/proxies can silently drop
// it. Polling /api/train/status guarantees the UI keeps updating metrics even
// if SSE dies. Only new events (by jobId + event count) are applied.

let lastPolledEvent = "";

export function pollStatus() {
  fetch("/api/train/status", { headers: { "X-API-Token": getTokenHeader() } })
    .then((r) => (r.ok ? r.json() : null))
    .then((s) => {
      if (!s) return;

      // Reflect paused state even if no new event arrived.
      if (s.paused !== store.paused) {
        set({ paused: s.paused, message: s.paused ? "Entrenamiento pausado" : "Entrenamiento reanudado" });
      }

      // Chart backfill: after a reload/restart the server may hold loss
      // history for the last job. Expose it so TrainView can repopulate.
      if (Array.isArray(s.loss_points) && s.loss_points.length !== store.lossPoints.length) {
        set({ lossPoints: s.loss_points });
      }

      const ev = s.last_event;
      if (!ev) return;

      const evJson = JSON.stringify(ev);
      if (evJson === lastPolledEvent) return;
      lastPolledEvent = evJson;

      handleTrainingEvent(ev);
    })
    .catch(() => {
      /* ignore transient errors */
    });
}

function getTokenHeader() {
  return localStorage.getItem("kuro_neko_token") || "";
}
