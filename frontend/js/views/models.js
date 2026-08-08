// Models view: list available model dirs with architecture details.

import { api } from "../core/api.js";
import { on } from "../core/events.js";
import { clear, fmtBytes, el } from "../core/dom.js";

export default class ModelsView {
  constructor(root) {
    this.root = root;
    this.models = [];
    on("job:done", () => this.render());
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("div", { class: "flex items-end justify-between mb-6" }, [
        el("div", {}, [
          el("h2", { class: "text-xl font-semibold text-white" }, "Modelos"),
          el("p", { class: "text-sm text-slate-400 mt-1" }, "Directorios con pesos de modelo cargables."),
        ]),
        el("button", {
          class: "px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm text-slate-200 transition",
          onclick: () => this.render(),
        }, "Refrescar"),
      ]),
    );

    try {
      this.models = await api.get("/api/models");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    if (this.models.length === 0) {
      this.root.append(
        el("div", { class: "p-10 rounded-2xl border border-dashed border-slate-700 text-center text-slate-500" },
          "Sin modelos. Entrena uno primero."),
      );
      return;
    }

    const grid = el("div", { class: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" });
    for (const m of this.models) {
      grid.append(this.modelCard(m));
    }
    this.root.append(grid);
  }

  modelCard(m) {
    const badge = m.default
      ? el("span", { class: "text-xs px-2 py-0.5 rounded-full bg-violet-900 text-violet-300" }, "activo")
      : null;

    return el("div", {
      class: "rounded-2xl border border-slate-800 bg-slate-900 p-5 hover:border-slate-600 transition",
    }, [
      el("div", { class: "flex items-center justify-between mb-3" }, [
        el("span", { class: "font-mono text-sm text-white" }, m.name),
        badge,
      ]),
      el("div", { class: "grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs text-slate-400" }, [
        el("span", {}, `d: ${m.n_embd}`),
        el("span", {}, `heads: ${m.n_head}`),
        el("span", {}, `layers: ${m.n_layer}`),
        el("span", {}, `ctx: ${m.n_ctx}`),
        el("span", {}, `vocab: ${m.vocab}`),
        el("span", {}, `${m.params_m.toFixed(2)} M params`),
      ]),
      el("div", { class: "mt-3 pt-3 border-t border-slate-800 flex justify-between text-xs text-slate-500" }, [
        el("span", {}, `peso ${fmtBytes(m.size_mb * 1024 * 1024)}`),
        el("span", { class: "text-slate-600 font-mono" }, m.dir),
      ]),
    ]);
  }
}
