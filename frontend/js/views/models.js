// Models view: browse model dirs, pick an active one, inspect and export.

import { api, getToken } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";
import { viewTitle, btn, badge, toast, iconSvg, stagger } from "../components/ui.js";

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
      viewTitle("Modelos", "Elige un modelo activo para Generar/Exportar, o exporta a GGUF directamente."),
    );

    try {
      this.models = await api.get("/api/models");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    if (this.models.length === 0) {
      this.root.append(
        el("div", { class: "p-10 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500" },
          "Sin modelos. Entrena uno primero."),
      );
      stagger(this.root);
      return;
    }

    this.root.append(this.activeBanner());

    const list = el("ul", { class: "mt-6 space-y-3" });
    for (const m of this.models) {
      list.append(this.row(m));
    }
    this.root.append(list);
    stagger(this.root);
  }

  activeBanner() {
    if (!this.active) return null;
    const m = this.models.find((x) => x.dir === this.active);
    if (!m) return null;
    return el("div", {
      class: "rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 flex items-center gap-3 shadow-lg shadow-emerald-500/10",
    }, [
      el("span", { class: "text-lg" }, "✓"),
      el("div", { class: "flex-1 min-w-0" }, [
        el("div", { class: "text-sm text-emerald-200 font-medium truncate" }, `Modelo activo: ${m.name}`),
        el("div", { class: "text-xs text-emerald-400/70 font-mono truncate" }, m.dir),
      ]),
      btn.ghost("Quitar", () => this.pick(null)),
    ]);
  }

  row(m) {
    const isActive = this.active === m.dir;

    const useBtn = el("button", {
      class: `px-3 py-1.5 rounded-lg text-sm font-medium transition active:scale-[0.98] ${
        isActive
          ? "bg-slate-800/70 text-slate-400 cursor-default border border-slate-700/60"
          : "text-white bg-emerald-600 hover:bg-emerald-500 shadow-lg shadow-emerald-600/20"
      }`,
      disabled: isActive || null,
      onclick: () => this.pick(m.dir),
    }, isActive ? "Activo ✓" : "Usar");

    const exportBtn = btn.primary(iconSvg("upload", "w-3.5 h-3.5") + " Exportar GGUF", () => this.exportGguf(m, exportBtn));
    exportBtn.classList.add("px-3", "py-1.5", "text-xs");

    return el("li", {
      class: "lift rounded-2xl border p-4 flex flex-wrap items-center gap-4 " +
        (isActive ? "border-emerald-500/40 bg-emerald-500/5" : "border-slate-800/70 bg-slate-900/40"),
    }, [
      el("div", { class: "flex-1 min-w-52" }, [
        el("div", { class: "flex items-center gap-2 flex-wrap" }, [
          el("span", { class: "font-mono text-sm text-white" }, m.name),
          m.default ? badge("default", "violet") : null,
          isActive ? badge("activo", "emerald", { pulse: true }) : null,
        ]),
        el("div", { class: "mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-500 tnum font-mono" }, [
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

  async exportGguf(m, trigger) {
    // Remove any previous export result card.
    this.root.querySelectorAll(".export-result").forEach((n) => n.remove());

    const btnNode = trigger || document.activeElement;
    const original = btnNode ? btnNode.innerHTML : "";
    if (btnNode) {
      btnNode.innerHTML = `${iconSvg("loader", "w-3.5 h-3.5 animate-spin")} Exportando…`;
      btnNode.disabled = true;
    }
    try {
      const res = await api.post("/api/export", { modelDir: m.dir, modelName: m.name });
      toast.success(`GGUF generado: ${res.gguf}`);
      const token = getToken();
      const url = `/api/download/${encodeURIComponent(res.gguf)}?token=${token}`;
      const done = el("div", {
        class: "export-result mt-3 p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-emerald-500/60 text-xs flex flex-wrap items-center gap-3",
      }, [
        el("span", { class: "text-emerald-300 font-medium flex-1 min-w-0 truncate font-mono" }, res.gguf),
        el("a", {
          href: url,
          class: "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition active:scale-[0.98] shrink-0",
          html: iconSvg("download", "w-3.5 h-3.5") + " Descargar GGUF",
        }),
      ]);
      this.root.append(done);
    } catch (e) {
      toast.error(`Error exportando: ${e.message}`);
    } finally {
      if (btnNode) {
        btnNode.innerHTML = original;
        btnNode.disabled = false;
      }
    }
  }
}
