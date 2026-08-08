// Models view: browse model dirs, pick an active one, inspect and export.

import { api, getToken } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";

const ACTIVE_KEY = "kuro_active_model";

export function getActiveModel() {
  return localStorage.getItem(ACTIVE_KEY) || "";
}

export function setActiveModel(dir) {
  localStorage.setItem(ACTIVE_KEY, dir);
  window.dispatchEvent(new CustomEvent("model:changed", { detail: dir }));
}

export default class ModelsView {
  constructor(root) {
    this.root = root;
    this.models = [];
    this.active = getActiveModel();
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("div", { class: "mb-6" }, [
        el("h2", { class: "text-xl font-semibold text-white" }, "Modelos"),
        el("p", { class: "text-sm text-slate-400 mt-1" }, "Elige un modelo activo para Generar/Exportar, o exporta a GGUF directamente."),
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

    this.root.append(this.activeBanner());

    const list = el("ul", { class: "mt-6 space-y-3" });
    for (const m of this.models) {
      list.append(this.row(m));
    }
    this.root.append(list);
  }

  activeBanner() {
    if (!this.active) return null;
    const m = this.models.find((x) => x.dir === this.active);
    if (!m) return null;
    return el("div", { class: "rounded-2xl border border-emerald-800 bg-emerald-950/30 p-4 flex items-center gap-3" }, [
      el("span", { class: "text-lg" }, "✓"),
      el("div", { class: "flex-1" }, [
        el("div", { class: "text-sm text-emerald-200 font-medium" }, `Modelo activo: ${m.name}`),
        el("div", { class: "text-xs text-emerald-400/70 font-mono" }, m.dir),
      ]),
      el("button", {
        class: "px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm text-slate-200 transition",
        onclick: () => this.pick(null),
      }, "Quitar"),
    ]);
  }

  row(m) {
    const isActive = this.active === m.dir;

    const useBtn = el("button", {
      class: `px-3 py-1.5 rounded-lg text-sm font-medium transition ${
        isActive ? "bg-slate-800 text-slate-400 cursor-default" : "bg-emerald-700 hover:bg-emerald-600 text-white"
      }`,
      disabled: isActive || null,
      onclick: () => this.pick(m.dir),
    }, isActive ? "Activo ✓" : "Usar");

    const exportBtn = el("button", {
      class: "px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition",
      onclick: () => this.exportGguf(m),
    }, "Exportar GGUF");

    return el("li", {
      class: "rounded-2xl border p-4 flex items-center gap-4 " +
        (isActive ? "border-emerald-700 bg-emerald-950/20" : "border-slate-800 bg-slate-900"),
    }, [
      el("div", { class: "flex-1 min-w-0" }, [
        el("div", { class: "flex items-center gap-2" }, [
          el("span", { class: "font-mono text-sm text-white" }, m.name),
          m.default
            ? el("span", { class: "text-[10px] px-1.5 py-0.5 rounded bg-violet-900 text-violet-200" }, "default")
            : null,
        ]),
        el("div", { class: "mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-500" }, [
          el("span", {}, `d ${m.n_embd}`),
          el("span", {}, `heads ${m.n_head}`),
          el("span", {}, `layers ${m.n_layer}`),
          el("span", {}, `ctx ${m.n_ctx}`),
          el("span", {}, `vocab ${m.vocab}`),
          el("span", {}, `${m.params_m.toFixed(2)}M`),
          el("span", {}, fmtBytes(m.size_mb * 1024 * 1024)),
        ]),
        el("div", { class: "mt-1 text-[10px] text-slate-600 font-mono truncate" }, m.dir),
      ]),
      el("div", { class: "flex gap-2 shrink-0" }, [useBtn, exportBtn]),
    ]);
  }

  pick(dir) {
    setActiveModel(dir || "");
    this.active = dir || "";
    this.render();
  }

  async exportGguf(m) {
    // Remove any previous export result card.
    this.root.querySelectorAll(".export-result").forEach((n) => n.remove());

    const btn = document.activeElement;
    const original = btn ? btn.textContent : "";
    if (btn) {
      btn.textContent = "Exportando…";
      btn.disabled = true;
    }
    try {
      const res = await api.post("/api/export", { modelDir: m.dir, modelName: m.name });
      const token = getToken();
      const url = `/api/download/${encodeURIComponent(res.gguf)}?token=${token}`;
      const done = el("div", { class: "export-result mt-3 p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs" }, [
        el("p", { class: "text-emerald-300 mb-2" }, `✓ GGUF generado: ${res.gguf}`),
        el("a", { href: url, class: "inline-block px-3 py-1.5 rounded bg-emerald-700 hover:bg-emerald-600 text-white" }, "⬇ Descargar GGUF"),
      ]);
      this.root.append(done);
    } catch (e) {
      alert(`Error exportando: ${e.message}`);
    } finally {
      if (btn) {
        btn.textContent = original;
        btn.disabled = false;
      }
    }
  }
}
