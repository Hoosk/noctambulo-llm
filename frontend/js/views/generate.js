// Generate view: run autoregressive text generation on a model.

import { api } from "../core/api.js";
import { clear, el } from "../core/dom.js";
import { getActiveModel } from "./models.js";

export default class GenerateView {
  constructor(root) {
    this.root = root;
    this.models = [];
    this.busy = false;
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("h2", { class: "text-xl font-semibold text-white mb-6" }, "Generar texto"),
    );

    try {
      this.models = await api.get("/api/models");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    const modelSel = el("select", { class: "w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white" });
    const active = getActiveModel();
    for (const m of this.models) {
      const selected = m.dir === active || (!active && m.default) || null;
      modelSel.append(el("option", { value: m.dir, selected }, `${m.name} · ${m.params_m.toFixed(1)}M`));
    }

    const prompt = el("textarea", {
      rows: 4,
      class: "w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white resize-y",
      placeholder: "Once upon a time...",
    }, "Once upon a time");

    const numInput = (id, val, min, step) =>
      el("input", {
        id,
        type: "number",
        value: val,
        min,
        step,
        class: "w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white",
      });

    const params = {
      max_tokens: numInput("gen-max", 60, 1, 1),
      temperature: numInput("gen-temp", 0.8, 0, 0.05),
      top_k: numInput("gen-topk", 40, 1, 1),
      top_p: numInput("gen-topp", 0.9, 0, 0.05),
      min_p: numInput("gen-minp", 0.05, 0, 0.01),
      repetition_penalty: numInput("gen-rep", 1.2, 1, 0.05),
    };

    const output = el("pre", { class: "mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 text-sm whitespace-pre-wrap text-slate-200 min-h-24" }, "—");
    const msg = el("p", { class: "text-sm text-slate-400 mt-3" }, "");

    const field = (label, input) =>
      el("label", { class: "block text-xs text-slate-400" }, [
        label,
        input,
      ]);

    const run = async () => {
      if (this.busy) return;
      this.busy = true;
      msg.textContent = "Generando…";
      msg.className = "text-sm text-violet-300 mt-3";
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
        output.textContent = res.text;
        msg.textContent = "";
      } catch (e) {
        output.textContent = "";
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-sm text-rose-400 mt-3";
      } finally {
        this.busy = false;
      }
    };

    const card = (title, body) =>
      el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
        el("h3", { class: "text-sm font-medium text-slate-300 mb-3" }, title),
        ...body,
      ]);

    const layout = el("div", { class: "grid grid-cols-1 lg:grid-cols-2 gap-5" }, [
      card("Configuración", [
        el("label", { class: "block text-xs text-slate-400 mb-3" }, ["Modelo", modelSel]),
        el("label", { class: "block text-xs text-slate-400 mb-3" }, ["Prompt", prompt]),
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
        el("button", {
          class: "mt-5 w-full py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium transition",
          onclick: run,
        }, "Generar"),
        msg,
      ]),
      card("Resultado", [output]),
    ]);

    this.root.append(layout);
  }
}
