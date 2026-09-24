// Export view: export a model dir to GGUF and offer download.

import { api, getToken } from "../core/api.js";
import { clear, el } from "../core/dom.js";
import { getActiveModel } from "./models.js";
import { viewTitle, btn, iconSvg, toast, stagger } from "../components/ui.js";

export default class ExportView {
  constructor(root) {
    this.root = root;
    this.models = [];
    this.busy = false;
  }

  async render() {
    clear(this.root);
    this.root.append(
      viewTitle("Exportar a GGUF", "Convierte un modelo entrenado al formato GGUF para usarlo en llama.cpp u otros runtimes."),
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

    const nameInput = el("input", {
      type: "text",
      value: "kuro-neko",
      class: inputCls,
    });

    const msg = el("p", { class: "text-xs text-slate-400 mt-3" }, "");
    const output = el("div", { class: "mt-4" }, "");

    const runBtn = btn.primary("Exportar", () => {});
    const run = async () => {
      if (this.busy) return;
      this.busy = true;
      runBtn.disabled = true;
      const prev = runBtn.innerHTML;
      runBtn.innerHTML = `${iconSvg("loader", "w-4 h-4 animate-spin")} Exportando…`;
      clear(output);
      try {
        const res = await api.post("/api/export", {
          modelDir: modelSel.value,
          modelName: nameInput.value || "kuro-neko",
        });
        const token = getToken();
        const dlUrl = `/api/download/${encodeURIComponent(res.gguf)}?token=${encodeURIComponent(token)}`;
        output.append(
          el("div", {
            class: "mt-1 p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-emerald-500/60 text-xs flex flex-wrap items-center gap-3",
          }, [
            el("span", { class: "text-emerald-300 font-medium flex-1 min-w-0 truncate font-mono" }, res.gguf),
            el("a", {
              href: dlUrl,
              class: "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition active:scale-[0.98] shrink-0",
              html: iconSvg("download", "w-3.5 h-3.5") + " Descargar GGUF",
            }),
          ]),
        );
        toast.success(`GGUF generado: ${res.gguf}`);
        msg.textContent = "";
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-xs text-rose-400 mt-3";
      } finally {
        this.busy = false;
        runBtn.disabled = false;
        runBtn.innerHTML = prev;
      }
    };
    runBtn.onclick = run;

    const card = el("div", { class: "max-w-md glass-card rounded-2xl p-5" }, [
      el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4" }, "Configuración"),
      el("label", { class: "block text-xs font-medium text-slate-400 mb-3" }, ["Modelo", modelSel]),
      el("label", { class: "block text-xs font-medium text-slate-400 mb-4" }, ["Nombre del GGUF", nameInput]),
      runBtn,
      msg,
      output,
    ]);

    this.root.append(card);
    stagger(this.root);
  }
}

const inputCls =
  "w-full px-3 py-2 mt-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition";
