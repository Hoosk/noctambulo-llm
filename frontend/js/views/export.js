// Export view: export a model dir to GGUF and offer download.

import { api, getToken } from "../core/api.js";
import { clear, el } from "../core/dom.js";

export default class ExportView {
  constructor(root) {
    this.root = root;
    this.models = [];
    this.busy = false;
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("h2", { class: "text-xl font-semibold text-white mb-6" }, "Exportar a GGUF"),
    );

    try {
      this.models = await api.get("/api/models");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    const modelSel = el("select", { class: "w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white" });
    for (const m of this.models) {
      modelSel.append(el("option", { value: m.dir, selected: m.default || null }, `${m.name} · ${m.params_m.toFixed(1)}M`));
    }

    const nameInput = el("input", {
      type: "text",
      value: "kuro-neko",
      class: "w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white",
    });

    const msg = el("p", { class: "text-sm text-slate-400 mt-3" }, "");
    const output = el("div", { class: "mt-4 text-sm text-slate-300" }, "");

    const run = async () => {
      if (this.busy) return;
      this.busy = true;
      msg.textContent = "Exportando…";
      msg.className = "text-sm text-violet-300 mt-3";
      clear(output);
      try {
        const res = await api.post("/api/export", {
          modelDir: modelSel.value,
          modelName: nameInput.value || "kuro-neko",
        });
        const token = getToken();
        const dlUrl = `/api/download/${encodeURIComponent(res.gguf)}?token=${encodeURIComponent(token)}`;
        output.append(
          el("p", { class: "mb-2" }, `GGUF generado: ${res.gguf}`),
          el("a", {
            href: dlUrl,
            class: "inline-block px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white font-medium transition",
          }, "⬇ Descargar GGUF"),
        );
        msg.textContent = "";
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-sm text-rose-400 mt-3";
      } finally {
        this.busy = false;
      }
    };

    const card = el("div", { class: "max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("label", { class: "block text-xs text-slate-400 mb-3" }, ["Modelo", modelSel]),
      el("label", { class: "block text-xs text-slate-400 mb-4" }, ["Nombre del GGUF", nameInput]),
      el("button", {
        class: "w-full py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium transition",
        onclick: run,
      }, "Exportar"),
      msg,
      output,
    ]);

    this.root.append(card);
  }
}
