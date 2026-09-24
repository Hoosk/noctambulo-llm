// Generate view: run autoregressive text generation on a model.

import { api } from "../core/api.js";
import { clear, el } from "../core/dom.js";
import { getActiveModel } from "./models.js";
import { viewTitle, btn, iconSvg, catLogoSvg, stagger } from "../components/ui.js";

export default class GenerateView {
  constructor(root) {
    this.root = root;
    this.models = [];
    this.busy = false;
  }

  async render() {
    clear(this.root);
    this.root.append(
      viewTitle("Generar texto", "Generación autorregresiva con parámetros de muestreo."),
    );

    try {
      this.models = await api.get("/api/models");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    const modelSel = el("select", { class: inputCls });
    const active = getActiveModel();
    for (const m of this.models) {
      const selected = m.dir === active || (!active && m.default) || null;
      modelSel.append(el("option", { value: m.dir, selected }, `${m.name} · ${m.params_m.toFixed(1)}M`));
    }

    const prompt = el("textarea", {
      rows: 4,
      class: `${inputCls} resize-y leading-relaxed`,
      placeholder: "Once upon a time...",
    }, "Once upon a time");

    const numInput = (id, val, min, step) =>
      el("input", {
        id,
        type: "number",
        value: val,
        min,
        step,
        class: inputCls,
      });

    const params = {
      max_tokens: numInput("gen-max", 60, 1, 1),
      temperature: numInput("gen-temp", 0.8, 0, 0.05),
      top_k: numInput("gen-topk", 40, 1, 1),
      top_p: numInput("gen-topp", 0.9, 0, 0.05),
      min_p: numInput("gen-minp", 0.05, 0, 0.01),
      repetition_penalty: numInput("gen-rep", 1.2, 1, 0.05),
    };

    const output = el("div", {
      class: "mt-4 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-violet-500/60 p-4 min-h-24",
    }, [
      el("pre", { class: "text-sm whitespace-pre-wrap text-slate-200 leading-relaxed" }, "—"),
    ]);
    const msg = el("p", { class: "text-xs text-slate-400 mt-3" }, "");

    const field = (label, input) =>
      el("label", { class: "block text-xs font-medium text-slate-400" }, [
        label,
        input,
      ]);

    const runBtn = btn.primary("Generar", () => {});
    const run = async () => {
      if (this.busy) return;
      this.busy = true;
      runBtn.disabled = true;
      const prev = runBtn.innerHTML;
      runBtn.innerHTML = `${iconSvg("loader", "w-4 h-4 animate-spin")} Generando…`;
      try {
        const res = await api.post("/api/generate", {
          modelDir: modelSel.value,
          prompt: prompt.value,
          maxTokens: Number(params.max_tokens.value),
          temperature: Number(params.temperature.value),
          topK: Number(params.top_k.value),
          topP: Number(params.top_p.value),
          minP: Number(params.min_p.value),
          repetitionPenalty: Number(params.repetition_penalty.value),
        });
        output.querySelector("pre").textContent = res.text;
        msg.textContent = "";
      } catch (e) {
        output.querySelector("pre").textContent = "";
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-xs text-rose-400 mt-3";
      } finally {
        this.busy = false;
        runBtn.disabled = false;
        runBtn.innerHTML = prev;
      }
    };
    runBtn.onclick = run;

    const card = (title, body, extra = "") =>
      el("div", { class: `glass-card rounded-2xl p-5 min-w-0 ${extra}` }, [
        el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4" }, title),
        ...[].concat(body),
      ]);

    const resultCard = card("Resultado", [
      el("div", { class: "flex items-start gap-3" }, [
        catLogoSvg("w-8 h-8 shrink-0 mt-1"),
        output,
      ]),
    ], "lg:sticky lg:top-6 self-start");

    const layout = el("div", { class: "grid grid-cols-1 lg:grid-cols-2 gap-5 items-start" }, [
      card("Configuración", [
        el("label", { class: "block text-xs font-medium text-slate-400 mb-3" }, ["Modelo", modelSel]),
        el("label", { class: "block text-xs font-medium text-slate-400 mb-3" }, ["Prompt", prompt]),
        el("div", { class: "grid grid-cols-3 gap-3 mt-2" }, [
          field("Max tokens", params.max_tokens),
          field("Temperatura", params.temperature),
          field("Top-K", params.top_k),
        ]),
        el("div", { class: "grid grid-cols-3 gap-3 mt-3" }, [
          field("Top-P", params.top_p),
          field("Min-P", params.min_p),
          field("Repetition", params.repetition_penalty),
        ]),
        el("div", { class: "mt-5" }, [runBtn]),
        msg,
      ]),
      resultCard,
    ]);

    this.root.append(layout);
    stagger(this.root);
  }
}

const inputCls =
  "w-full px-3 py-2 mt-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition";
