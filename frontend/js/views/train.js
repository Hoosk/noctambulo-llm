// Train view: configure + launch training, watch live metrics.

import { api } from "../core/api.js";
import { on } from "../core/events.js";
import { getJob, JobState } from "../core/store.js";
import { clear, el, fmtBytes, fmtDur } from "../core/dom.js";
import LossChart from "../components/chart.js";
import { statCard, stagger } from "../components/ui.js";

export default class TrainView {
  constructor(root) {
    this.root = root;
    this.chart = new LossChart();
    this.fields = {};
    this.busy = false;

    on("job:changed", () => this.sync());

    this.logSince = 0;
    this.logLines = [];
    this.logTimer = null;
    this._logBusy = false;
    this._epochsSynced = false;
  }

  async render() {
    clear(this.root);
    this.chart.destroy();
    const [datasets, devices] = await Promise.allSettled([
      api.get("/api/datasets"),
      api.get("/api/devices"),
    ]);

    const left = this.buildForm(datasets.status === "fulfilled" ? datasets.value : [], devices.status === "fulfilled" ? devices.value : []);
    const right = this.buildMetrics();

    this.root.append(
      el("div", { class: "grid min-w-0 grid-cols-1 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] gap-6 items-start" }, [
        left,
        right,
      ]),
    );
    stagger(this.root);

    // Chart.js must init after the canvas is in the DOM.
    this.chart.attach();
    const job = getJob();
    if (job.lossPoints && job.lossPoints.length > 0) {
      this.chart.setHistory(job.lossPoints);
    }

    // Reset the log cursor so re-rendering the view re-fetches the server's
    // kept lines instead of showing only the delta since the previous visit.
    this.logSince = 0;
    this.logLines = [];
    this.sync();
    this.startLogPolling();
  }

  startLogPolling() {
    if (this.logTimer) return;
    this.logTimer = setInterval(() => this.tickLogs(), 1000);
    this.tickLogs();
  }

  async tickLogs() {
    const view = document.getElementById("train-log");
    if (!view || this._logBusy) return;
    this._logBusy = true;
    try {
      const res = await api.getLogs(this.logSince);
      if (res && Array.isArray(res.lines) && res.lines.length > 0) {
        for (const l of res.lines) {
          this.logLines.push(l.line);
        }
        // Cap client-side lines so long sessions cannot grow the DOM forever.
        if (this.logLines.length > 800) {
          this.logLines.splice(0, this.logLines.length - 800);
        }
        this.logSince = res.total;
        this.renderLogLines(view);
      }
    } catch {
      /* ignore transient errors */
    } finally {
      this._logBusy = false;
    }
  }

  // Terminal-style render: color each line by severity keyword.
  renderLogLines(view) {
    const lineNode = (line) => {
      let cls = "text-slate-400";
      if (/error|panic|failed|fatal/i.test(line)) cls = "text-rose-400";
      else if (/warn/i.test(line)) cls = "text-amber-300";
      else if (/^== |checkpoint/i.test(line)) cls = "text-violet-300";
      return el("div", { class: cls }, line);
    };
    view.replaceChildren(...this.logLines.map(lineNode));
    view.scrollTop = view.scrollHeight;
  }

  // ------------------------------------------------------------------ form
  buildForm(datasets, devices) {
    const dsSel = el("select", { id: "train-dataset", class: inputCls });
    if (datasets.length === 0) {
      dsSel.append(el("option", { value: "" }, "sin datasets"));
    } else {
      for (const d of datasets) dsSel.append(el("option", { value: d.name }, d.name));
    }
    this.fields["train-dataset"] = dsSel;

    const devSel = el("select", { id: "train-device", class: inputCls });
    for (const d of devices) {
      const sel = d.selector || d.name;
      devSel.append(el("option", { value: sel, selected: sel === "DefaultDevice" || null }, sel));
    }
    this.fields["train-device"] = devSel;

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
      return el("label", { class: fieldLabelCls }, [label, input]);
    };

    const textField = (label, id, val) => {
      const input = el("input", { id, type: "text", value: val, class: inputCls });
      this.fields[id] = input;
      return el("label", { class: fieldLabelCls }, [label, input]);
    };

    const startBtn = el("button", {
      id: "train-start",
      class: "flex-1 py-2 rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium shadow-lg shadow-violet-600/25 transition active:scale-[0.98] disabled:opacity-40",
      onclick: () => this.start(),
    }, "Iniciar");

    const stopBtn = el("button", {
      id: "train-stop",
      class: "flex-1 py-2 rounded-lg bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white text-sm font-medium shadow-lg shadow-rose-600/20 transition active:scale-[0.98] disabled:opacity-40",
      onclick: () => this.stop(),
    }, "Detener");

    const pauseBtn = el("button", {
      id: "train-pause",
      class: "flex-1 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-sm font-medium shadow-lg shadow-amber-600/20 transition active:scale-[0.98] disabled:opacity-40",
      onclick: () => this.pause(),
    }, "Pausar");

    const msg = el("p", { id: "train-msg", class: "text-xs text-slate-400 mt-3" }, "");

    const resumeCheckbox = el("input", {
      id: "train-resume",
      type: "checkbox",
      class: "w-4 h-4 rounded accent-violet-500 cursor-pointer",
      onchange: (e) => this.toggleResume(e.target.checked),
    });

    const resumeSelect = el("select", {
      id: "train-resume-cp",
      class: `${inputCls} disabled:opacity-40 disabled:cursor-not-allowed`,
      disabled: true,
    }, el("option", { value: "" }, "— selecciona checkpoint —"));

    const card = el("div", { class: "glass-card rounded-2xl p-5 lg:sticky lg:top-6 self-start min-w-0" }, [
      el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4" }, "Configuración"),
      el("label", { class: fieldLabelCls + " mb-3" }, ["Dataset", dsSel]),
      el("label", { class: fieldLabelCls + " mb-4" }, ["Dispositivo", devSel]),
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
        field("KV heads (GQA)", "train-nKvHead", 0, { min: 0 }),
        field("Dropout", "train-dropout", 0.1, { step: 0.05, min: 0, max: 0.9 }),
        field("Attn softcap", "train-attnSoftcap", 0, { step: 1, min: 0 }),
        field("Seq warmup steps", "train-seqWarmup", 0, { min: 0 }),
      ]),
      el("div", { class: "grid grid-cols-2 gap-3 mt-3" }, [
        textField("Save dir", "train-saveDir", "checkpoints_web"),
        textField("Output dir", "train-outputDir", "model_output"),
      ]),
      el("div", { class: "mt-4" }, [
        el("label", { class: "flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none" }, [
          resumeCheckbox,
          "Reanudar pesos desde checkpoint (warm start)",
        ]),
        resumeSelect,
      ]),
      el("div", { class: "flex gap-3 mt-5" }, [startBtn, pauseBtn, stopBtn]),
      msg,
    ]);

    this.fields["train-resume"] = resumeCheckbox;
    this.fields["train-resume-cp"] = resumeSelect;

    const saveDirField = this.fields["train-saveDir"];
    if (saveDirField) {
      saveDirField.addEventListener("change", () => {
        if (this.fields["train-resume"]?.checked) {
          this.refreshCheckpoints();
        }
      });
    }

    return card;
  }

  // Enable/disable the checkpoint selector and load the file list from the
  // "Save dir" field when the user ticks "Reanudar desde checkpoint".
  async toggleResume(enabled) {
    const select = this.fields["train-resume-cp"];
    if (!select) return;
    select.disabled = !enabled;
    if (enabled) {
      await this.refreshCheckpoints();
    }
  }

  async refreshCheckpoints() {
    const select = this.fields["train-resume-cp"];
    const saveDir = this.fields["train-saveDir"]?.value || "checkpoints_web";
    if (!select) return;
    select.innerHTML = "";
    select.append(el("option", { value: "" }, "— selecciona checkpoint —"));
    try {
      const cps = await api.getCheckpoints(saveDir);
      if (!cps || cps.length === 0) {
        select.append(el("option", { value: "" }, "sin checkpoints en " + saveDir));
        select.disabled = true;
        return;
      }
      for (const cp of cps) {
        select.append(el("option", { value: cp.path }, `${cp.name} · ${fmtBytes(cp.size_bytes)}`));
      }
      select.disabled = false;
    } catch (e) {
      select.append(el("option", { value: "" }, `error: ${e.message}`));
      select.disabled = true;
    }
  }

  // --------------------------------------------------------------- metrics
  buildMetrics() {
    const statCards = el("div", { id: "train-stats", class: "grid min-w-0 grid-cols-2 sm:grid-cols-3 gap-3 mb-5" });

    const card = el("div", { class: "glass-card rounded-2xl p-5 min-w-0" }, [
      el("div", { class: "flex items-center justify-between gap-3 mb-5" }, [
        el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500" }, "Métricas en vivo"),
        el("span", { id: "train-status-badge", class: badgeIdleCls }, "idle"),
      ]),
      statCards,
      this.chart.build(),
      el("div", { class: "mt-5" }, [
        el("h4", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2" }, "Progress"),
        el("div", { class: "h-2.5 rounded-full bg-slate-800/80 overflow-hidden" }, [
          el("div", { id: "train-progress", class: "h-full w-0 bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300" }),
        ]),
      ]),
      el("div", { class: "mt-5" }, [
        el("h4", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2" }, "Preview"),
        el("pre", { id: "train-preview", class: "p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-violet-500/60 text-xs whitespace-pre-wrap text-slate-300 min-h-16 leading-relaxed" }, "—"),
      ]),
      el("details", { class: "mt-5 group", open: true }, [
        el("summary", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 cursor-pointer select-none hover:text-slate-300 transition" }, "Logs del worker"),
        el("pre", { id: "train-log", class: "mt-2 p-3 rounded-xl bg-black/40 border border-slate-800/70 text-[11px] font-mono leading-relaxed whitespace-pre-wrap h-44 overflow-y-auto" }, ""),
      ]),
    ]);

    return card;
  }

  // --------------------------------------------------------------- actions
  async start() {
    if (this.busy) return;
    const msg = document.getElementById("train-msg");

    const read = (id) => {
      const field = this.fields[id];
      if (!field) throw new Error(`Campo de configuración ausente: ${id}`);
      return field;
    };

    let body;
    try {
      const resumeOn = this.fields["train-resume"]?.checked;
      const resumePath = this.fields["train-resume-cp"]?.value || "";
      body = {
        file: read("train-dataset").value,
        device: read("train-device").value,
        epochs: Number(read("train-epochs").value),
        batchSize: Number(read("train-batch").value),
        lr: Number(read("train-lr").value),
        maxTokens: Number(read("train-maxTokens").value) || null,
        nCtx: Number(read("train-nCtx").value),
        nEmbd: Number(read("train-nEmbd").value),
        nHead: Number(read("train-nHead").value),
        nLayer: Number(read("train-nLayer").value),
        valSplit: Number(read("train-valSplit").value),
        gradAccum: Number(read("train-gradAccum").value),
        nKvHead: Number(read("train-nKvHead").value) || null,
        dropout: Number(read("train-dropout").value),
        attnSoftcap: Number(read("train-attnSoftcap").value),
        seqWarmupSteps: Number(read("train-seqWarmup").value),
        saveDir: read("train-saveDir").value,
        outputDir: read("train-outputDir").value,
        resume: resumeOn && resumePath ? resumePath : undefined,
      };
    } catch (e) {
      msg.textContent = `Error de configuración: ${e.message}`;
      msg.className = "text-xs text-rose-400 mt-3";
      return;
    }

    if (!body.file || body.file === "sin datasets") {
      msg.textContent = "Selecciona un dataset para entrenar.";
      msg.className = "text-xs text-rose-400 mt-3";
      return;
    }

    // Resume checked but no checkpoint selected: fail loudly instead of
    // silently training from scratch.
    const resumeOn = this.fields["train-resume"]?.checked;
    const resumePath = this.fields["train-resume-cp"]?.value || "";
    if (resumeOn && !resumePath) {
      msg.textContent = "Selecciona un checkpoint para reanudar o desactiva la casilla.";
      msg.className = "text-xs text-rose-400 mt-3";
      return;
    }

    this.busy = true;
    this.chart.reset();
    msg.textContent = "Iniciando…";
    msg.className = "text-xs text-violet-300 mt-3";
    try {
      const res = await api.post("/api/train", body);
      msg.textContent = res.jobId ? `Entrenamiento iniciado (${res.jobId})` : "Entrenamiento iniciado";
      msg.className = "text-xs text-emerald-300 mt-3";
      // Reset the resume toggle so the next run does not silently reuse it.
      if (this.fields["train-resume"]) this.fields["train-resume"].checked = false;
      const cpSel = this.fields["train-resume-cp"];
      if (cpSel) {
        cpSel.value = "";
        cpSel.disabled = true;
      }
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-xs text-rose-400 mt-3";
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
      msg.className = "text-xs text-amber-300 mt-3";
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-xs text-rose-400 mt-3";
    }
  }

  async pause() {
    const msg = document.getElementById("train-msg");
    try {
      await api.post("/api/train/pause", {});
      msg.textContent = "Pausando…";
      msg.className = "text-xs text-amber-300 mt-3";
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-xs text-rose-400 mt-3";
    }
  }

  async resume() {
    const msg = document.getElementById("train-msg");
    try {
      await api.post("/api/train/resume", {});
      msg.textContent = "Reanudando…";
      msg.className = "text-xs text-emerald-300 mt-3";
    } catch (e) {
      msg.textContent = `Error: ${e.message}`;
      msg.className = "text-xs text-rose-400 mt-3";
    }
  }

  // ------------------------------------------------------------ live sync
  sync() {
    const job = getJob();
    const badge = document.getElementById("train-status-badge");
    const startBtn = document.getElementById("train-start");
    const stopBtn = document.getElementById("train-stop");
    const pauseBtn = document.getElementById("train-pause");

    // Reflect the running job's epoch count in the form, but only once per run
    // so the user can still edit it for the *next* run while one is active.
    const epochsField = this.fields["train-epochs"];
    if (epochsField) {
      if (job.status === JobState.RUNNING && job.epochs > 0 && !this._epochsSynced) {
        epochsField.value = String(job.epochs);
        this._epochsSynced = true;
      } else if (job.status !== JobState.RUNNING) {
        this._epochsSynced = false;
      }
    }

    if (badge) {
      const map = {
        [JobState.RUNNING]: ["border-violet-500/30 bg-violet-500/15 text-violet-300", "entrenando", true],
        [JobState.DONE]: ["border-emerald-500/30 bg-emerald-500/15 text-emerald-300", "completado", false],
        [JobState.STOPPED]: ["border-amber-500/30 bg-amber-500/15 text-amber-300", "detenido", false],
        [JobState.ERROR]: ["border-rose-500/30 bg-rose-500/15 text-rose-300", "error", false],
      };
      const running = job.status === JobState.RUNNING;
      const [cls, label, pulse] = map[job.status] || [badgeIdleCls, "idle", false];
      badge.className = `inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full border ${cls}${pulse ? " animate-pulse" : ""}`;
      badge.textContent = job.paused && running ? "pausado" : job.stopping && running ? "deteniendo" : label;
    }
    if (startBtn) startBtn.disabled = job.status === JobState.RUNNING;
    if (stopBtn) {
      stopBtn.disabled = job.status !== JobState.RUNNING || job.stopping;
      stopBtn.textContent = job.stopping ? "Deteniendo…" : "Detener";
    }
    if (pauseBtn) {
      const running = job.status === JobState.RUNNING;
      pauseBtn.disabled = !running || job.stopping;
      pauseBtn.textContent = job.paused ? "Reanudar" : "Pausar";
      pauseBtn.className = `flex-1 py-2 rounded-lg text-white text-sm font-medium shadow-lg transition active:scale-[0.98] disabled:opacity-40 ${
        job.paused
          ? "bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/20"
          : "bg-amber-600 hover:bg-amber-500 shadow-amber-600/20"
      }`;
      pauseBtn.onclick = job.paused ? () => this.resume() : () => this.pause();
    }

    const pb = document.getElementById("train-progress");
    if (pb) {
      pb.style.width = `${job.pct || 0}%`;
      pb.classList.toggle("progress-shimmer", job.status === JobState.RUNNING && !job.paused);
    }

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

    const stat = (label, value, iconName, tone = "slate", sub) =>
      statCard(label, value, { icon: iconName, tone, sub });

    const lossGood = job.loss != null && job.avgLoss != null && job.loss <= job.avgLoss;

    const items = [
      stat("Epoch", job.epochs ? `${job.epoch}/${job.epochs}` : "—", "layers", "violet"),
      stat("Batch", job.numBatches ? `${job.batch}/${job.numBatches}` : "—", "gauge", "sky"),
      stat("Contexto", job.nCtx ? `${job.nCtx} tok` : "—", "cpu", "sky"),
      stat("Loss", job.loss != null ? job.loss.toFixed(4) : "—", "activity", job.loss == null ? "slate" : lossGood ? "emerald" : "amber"),
      stat("Avg Loss", job.avgLoss != null ? job.avgLoss.toFixed(4) : "—", "gauge", "slate"),
      stat("PPL val", job.valPpl != null ? job.valPpl.toFixed(2) : "—", "zap", "fuchsia"),
      stat("Velocidad", job.batchesPerSec != null ? `${job.batchesPerSec.toFixed(1)} b/s` : "—", "zap", "sky"),
      stat("ETA", fmtDur(job.etaSecs), "clock", "amber"),
      stat("Transcurrido", fmtDur(job.elapsedSecs), "clock", "slate"),
      stat("Progreso", job.pct != null ? `${job.pct.toFixed(1)}%` : "—", "gauge", "violet"),
    ];
    for (const it of items) box.append(it);
  }
}

const badgeIdleCls = "inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full border border-slate-700 bg-slate-800/80 text-slate-400";
const fieldLabelCls = "block text-xs font-medium text-slate-400";
const inputCls =
  "w-full px-3 py-2 mt-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition";
