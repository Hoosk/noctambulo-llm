// Train view: configure + launch training, watch live metrics.

import { api } from "../core/api.js";
import { on } from "../core/events.js";
import { getJob, JobState } from "../core/store.js";
import { clear, el, fmtBytes, fmtDur } from "../core/dom.js";
import LossChart from "../components/chart.js";

export default class TrainView {
  constructor(root) {
    this.root = root;
    this.chart = new LossChart();
    this.fields = {};
    this.busy = false;

    on("job:changed", () => this.sync());
  }

  async render() {
    clear(this.root);
    const [datasets, devices] = await Promise.allSettled([
      api.get("/api/datasets"),
      api.get("/api/devices"),
    ]);

    const left = this.buildForm(datasets.status === "fulfilled" ? datasets.value : [], devices.status === "fulfilled" ? devices.value : []);
    const right = this.buildMetrics();

    this.root.append(
      el("div", { class: "grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6" }, [
        left,
        right,
      ]),
    );

    this.sync();
  }

  // ------------------------------------------------------------------ form
  buildForm(datasets, devices) {
    const dsSel = el("select", { id: "train-dataset", class: inputCls });
    if (datasets.length === 0) {
      dsSel.append(el("option", {}, "sin datasets"));
    } else {
      for (const d of datasets) dsSel.append(el("option", { value: d }, d));
    }

    const devSel = el("select", { id: "train-device", class: inputCls });
    for (const d of devices) {
      devSel.append(el("option", { value: d.name, selected: d.name === "DefaultDevice" || null }, d.name));
    }

    const field = (label, id, val, opts = {}) => {
      const input = el("input", {
        id,
        type: opts.type || "number",
        value: val,
        ...(opts.step ? { step: opts.step } : {}),
        ...(opts.min != null ? { min: opts.min } : {}),
        class: inputCls,
      });
      this.fields[id] = input;
      return el("label", { class: "block text-xs text-slate-400" }, [label, input]);
    };

    const textField = (label, id, val) => {
      const input = el("input", { id, type: "text", value: val, class: inputCls });
      this.fields[id] = input;
      return el("label", { class: "block text-xs text-slate-400" }, [label, input]);
    };

    const startBtn = el("button", {
      id: "train-start",
      class: "flex-1 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium transition disabled:opacity-40",
      onclick: () => this.start(),
    }, "Iniciar");

    const stopBtn = el("button", {
      id: "train-stop",
      class: "flex-1 py-2 rounded-lg bg-rose-700 hover:bg-rose-600 text-white font-medium transition disabled:opacity-40",
      onclick: () => this.stop(),
    }, "Detener");

    const msg = el("p", { id: "train-msg", class: "text-sm text-slate-400 mt-3" }, "");

    const card = el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("h3", { class: "text-sm font-medium text-slate-300 mb-4" }, "Configuración"),
      el("label", { class: "block text-xs text-slate-400 mb-3" }, ["Dataset", dsSel]),
      el("label", { class: "block text-xs text-slate-400 mb-4" }, ["Dispositivo", devSel]),
      el("div", { class: "grid grid-cols-2 gap-3" }, [
        field("Epochs", "train-epochs", 1, { min: 1 }),
        field("Batch", "train-batch", 32, { min: 1 }),
        field("LR", "train-lr", 0.00005, { step: 0.00001, min: 0 }),
        field("Max tokens", "train-maxTokens", 10000000, { min: 0 }),
        field("n_ctx", "train-nCtx", 256, { min: 16 }),
        field("n_embd", "train-nEmbd", 256, { min: 16 }),
        field("n_head", "train-nHead", 4, { min: 1 }),
        field("n_layer", "train-nLayer", 4, { min: 1 }),
        field("Val split", "train-valSplit", 0.02, { step: 0.01, min: 0, max: 0.9 }),
        field("Grad accum", "train-gradAccum", 1, { min: 1 }),
      ]),
      el("div", { class: "grid grid-cols-2 gap-3 mt-3" }, [
        textField("Save dir", "train-saveDir", "checkpoints_web"),
        textField("Output dir", "train-outputDir", "model_output"),
      ]),
      el("div", { class: "flex gap-3 mt-5" }, [startBtn, stopBtn]),
      msg,
    ]);

    return card;
  }

  // --------------------------------------------------------------- metrics
  buildMetrics() {
    const statCards = el("div", { id: "train-stats", class: "grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4" });

    const card = el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("div", { class: "flex items-center justify-between mb-4" }, [
        el("h3", { class: "text-sm font-medium text-slate-300" }, "Métricas en vivo"),
        el("span", { id: "train-status-badge", class: "text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400" }, "idle"),
      ]),
      statCards,
      this.chart.build(),
      el("div", { class: "mt-4" }, [
        el("h4", { class: "text-xs font-medium text-slate-400 mb-2" }, "Progress"),
        el("div", { class: "h-2 rounded-full bg-slate-800 overflow-hidden" }, [
          el("div", { id: "train-progress", class: "h-full bg-violet-600 transition-all duration-300", style: "width:0%" }),
        ]),
      ]),
      el("div", { class: "mt-5" }, [
        el("h4", { class: "text-xs font-medium text-slate-400 mb-2" }, "Preview"),
        el("pre", { id: "train-preview", class: "p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs whitespace-pre-wrap text-slate-300 min-h-16" }, "—"),
      ]),
    ]);

    return card;
  }

  // --------------------------------------------------------------- actions
  async start() {
    if (this.busy) return;
    const body = {
      file: this.fields["train-dataset"].value,
      device: this.fields["train-device"].value,
      epochs: Number(this.fields["train-epochs"].value),
      batchSize: Number(this.fields["train-batch"].value),
      lr: Number(this.fields["train-lr"].value),
      maxTokens: Number(this.fields["train-maxTokens"].value) || null,
      nCtx: Number(this.fields["train-nCtx"].value),
      nEmbd: Number(this.fields["train-nEmbd"].value),
      nHead: Number(this.fields["train-nHead"].value),
      nLayer: Number(this.fields["train-nLayer"].value),
      valSplit: Number(this.fields["train-valSplit"].value),
      gradAccum: Number(this.fields["train-gradAccum"].value),
      saveDir: this.fields["train-saveDir"].value,
      outputDir: this.fields["train-outputDir"].value,
    };

    const msg = document.getElementById("train-msg");
    this.busy = true;
    msg.textContent = "Iniciando…";
    msg.className = "text-sm text-violet-300 mt-3";
    try {
      const res = await api.post("/api/train", body);
      this.chart.reset();
      msg.textContent = res.jobId ? `Entrenamiento iniciado (${res.jobId})` : "Entrenamiento iniciado";
      msg.className = "text-sm text-emerald-300 mt-3";
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-sm text-rose-400 mt-3";
    } finally {
      this.busy = false;
      this.sync();
    }
  }

  async stop() {
    const msg = document.getElementById("train-msg");
    try {
      await api.post("/api/train/stop", {});
      msg.textContent = "Parando…";
      msg.className = "text-sm text-amber-300 mt-3";
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-sm text-rose-400 mt-3";
    }
  }

  // ------------------------------------------------------------ live sync
  sync() {
    const job = getJob();
    const badge = document.getElementById("train-status-badge");
    const startBtn = document.getElementById("train-start");
    const stopBtn = document.getElementById("train-stop");

    if (badge) {
      const map = {
        [JobState.RUNNING]: ["bg-violet-900 text-violet-300", "entrenando"],
        [JobState.DONE]: ["bg-emerald-900 text-emerald-300", "completado"],
        [JobState.STOPPED]: ["bg-amber-900 text-amber-300", "detenido"],
        [JobState.ERROR]: ["bg-rose-900 text-rose-300", "error"],
      };
      const [cls, label] = map[job.status] || ["bg-slate-800 text-slate-400", "idle"];
      badge.className = `text-xs px-2 py-0.5 rounded-full ${cls}`;
      badge.textContent = label;
    }
    if (startBtn) startBtn.disabled = job.status === JobState.RUNNING;
    if (stopBtn) stopBtn.disabled = job.status !== JobState.RUNNING;

    const pb = document.getElementById("train-progress");
    if (pb) pb.style.width = `${job.pct || 0}%`;

    const preview = document.getElementById("train-preview");
    if (preview && job.preview) preview.textContent = job.preview;

    this.renderStats(job);
    if (job.loss != null) {
      this.chart.addPoint(job.step || 0, job.loss);
    }
  }

  renderStats(job) {
    const box = document.getElementById("train-stats");
    if (!box) return;
    clear(box);

    const stat = (label, value, sub) =>
      el("div", { class: "rounded-xl bg-slate-950 border border-slate-800 px-4 py-3" }, [
        el("div", { class: "text-xs text-slate-500" }, label),
        el("div", { class: "text-lg font-semibold text-white mt-0.5" }, value),
        sub ? el("div", { class: "text-xs text-slate-500 mt-0.5" }, sub) : null,
      ]);

    const items = [
      stat("Epoch", job.epochs ? `${job.epoch}/${job.epochs}` : "—"),
      stat("Batch", job.numBatches ? `${job.batch}/${job.numBatches}` : "—"),
      stat("Loss", job.loss != null ? job.loss.toFixed(4) : "—"),
      stat("Avg Loss", job.avgLoss != null ? job.avgLoss.toFixed(4) : "—"),
      stat("PPL val", job.valPpl != null ? job.valPpl.toFixed(2) : "—"),
      stat("Velocidad", job.batchesPerSec != null ? `${job.batchesPerSec.toFixed(1)} b/s` : "—"),
      stat("ETA", fmtDur(job.etaSecs)),
      stat("Transcurrido", fmtDur(job.elapsedSecs)),
      stat("Progreso", job.pct != null ? `${job.pct.toFixed(1)}%` : "—"),
    ];
    for (const it of items) box.append(it);
  }
}

const inputCls =
  "w-full px-3 py-2 mt-1 rounded-lg bg-slate-950 border border-slate-700 text-white focus:outline-none focus:ring-2 focus:ring-violet-500";
