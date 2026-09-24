// Datasets view: download from HF, organize local corpus, combine with weights.

import { api } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";
import { viewTitle, btn, iconSvg, toast, modal, stagger } from "../components/ui.js";

export default class DatasetsView {
  constructor(root) {
    this.root = root;
    this.datasets = [];
    this.selected = new Set();
  }

  async render() {
    clear(this.root);
    this.root.append(
      viewTitle("Datasets", "Descarga, organiza y combina corpus.", [
        btn.ghost(iconSvg("refresh", "w-3.5 h-3.5") + " Refrescar", () => this.render()),
      ]),
    );

    this.root.append(this.downloadCard());
    this.root.append(el("div", { class: "h-6" }));
    this.root.append(this.searchCard());

    await this.loadList();
    stagger(this.root);
  }

  // ------------------------------------------------------------- search HF
  searchCard() {
    const searchInput = el("input", {
      type: "text",
      placeholder: "Buscar datasets en Hugging Face (ej. cosmopedia, stories, code)...",
      class: `w-full pl-9 pr-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition`,
    });
    const msg = el("p", { class: "text-xs text-slate-400 mt-3" }, "");
    const results = el("div", { class: "mt-3 space-y-2" });

    const search = async () => {
      const q = searchInput.value.trim();
      if (!q) return;
      clear(results);
      msg.textContent = "Buscando…";
      msg.className = "text-xs text-violet-300 mt-3";
      try {
        const hits = await api.get(`/api/datasets/search?q=${encodeURIComponent(q)}`);
        msg.textContent = "";
        if (hits.length === 0) {
          results.append(el("p", { class: "text-xs text-slate-500" }, "Sin resultados."));
          return;
        }
        for (const h of hits) {
          results.append(this.searchRow(h));
        }
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-xs text-rose-400 mt-3";
      }
    };

    searchInput.addEventListener("keydown", (e) => e.key === "Enter" && search());

    return el("div", { class: "glass-card rounded-2xl p-5" }, [
      el("div", { class: "flex items-center justify-between mb-3" }, [
        el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500" }, "Buscar en Hugging Face"),
      ]),
      el("div", { class: "flex gap-3 items-start" }, [
        el("div", { class: "relative flex-1" }, [
          el("span", { class: "absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" }, iconSvg("search", "w-4 h-4")),
          searchInput,
        ]),
        btn.primary("Buscar", search),
      ]),
      msg,
      results,
    ]);
  }

  searchRow(h) {
    const lang = h.tags && h.tags.length
      ? badge(h.tags[0].replace("language:", ""), "sky")
      : null;

    const dl = async () => {
      try {
        await api.post("/api/datasets/download", { repoId: h.id });
        toast.success(`Descargando ${h.id}…`);
        this.loadList();
      } catch (e) {
        toast.error(`Error descargando: ${e.message}`);
      }
    };

    return el("div", {
      class: "lift rounded-xl bg-slate-950/70 border border-slate-800/70 px-4 py-3 flex items-center gap-3",
    }, [
      el("div", { class: "flex-1 min-w-0" }, [
        el("div", { class: "flex items-center gap-2 flex-wrap" }, [
          el("span", { class: "text-sm text-slate-200 font-mono truncate" }, h.id),
          lang,
        ]),
        el("div", { class: "text-xs text-slate-500 mt-0.5 tnum" }, [
          `${h.downloads.toLocaleString("es-ES")} descargas · ${h.likes.toLocaleString("es-ES")} likes`,
        ]),
      ]),
      btn.primary(iconSvg("download", "w-3.5 h-3.5") + " Descargar", dl, "px-3 py-1.5 text-xs shrink-0"),
    ]);
  }

  // ------------------------------------------------------------- download
  downloadCard() {
    const repoInput = el("input", {
      type: "text",
      placeholder: "HuggingFaceTB/cosmopedia-100k",
      class: inputBaseCls,
    });
    const fileInput = el("input", {
      type: "text",
      placeholder: "archivo dentro del repo (opcional)",
      class: inputBaseCls,
    });
    const msg = el("p", { class: "text-xs text-slate-400 mt-3" }, "");

    const download = async () => {
      const repoId = repoInput.value.trim();
      if (!repoId) {
        msg.textContent = "Indica un repo de Hugging Face.";
        msg.className = "text-xs text-rose-400 mt-3";
        return;
      }
      msg.textContent = "Descargando… (puede tardar)";
      msg.className = "text-xs text-violet-300 mt-3";
      try {
        const res = await api.post("/api/datasets/download", {
          repoId,
          filename: fileInput.value.trim() || "TinyStories-train.txt",
        });
        msg.textContent = `Descargado: ${res.name}`;
        msg.className = "text-xs text-emerald-300 mt-3";
        toast.success(`Descargado: ${res.name}`);
        this.loadList();
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-xs text-rose-400 mt-3";
      }
    };

    return el("div", { class: "glass-card rounded-2xl p-5" }, [
      el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4" }, "Descargar de Hugging Face"),
      el("div", { class: "grid grid-cols-1 sm:grid-cols-2 gap-3" }, [
        el("label", { class: "block text-xs font-medium text-slate-400" }, ["Repo ID", repoInput]),
        el("label", { class: "block text-xs font-medium text-slate-400" }, ["Archivo (opcional)", fileInput]),
      ]),
      el("div", { class: "mt-4" }, [
        btn.primary(iconSvg("download", "w-3.5 h-3.5") + " Descargar", download),
      ]),
      msg,
    ]);
  }

  // ------------------------------------------------------------- listing
  async loadList() {
    // Drop previous list/combine nodes so repeated refreshes never stack them.
    this.root.querySelectorAll(".datasets-list-host, .datasets-combine-card").forEach((n) => n.remove());

    const host = document.createElement("div");
    host.className = "datasets-list-host";
    try {
      this.datasets = await api.get("/api/datasets");
    } catch (e) {
      host.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      this.root.append(host);
      return;
    }

    // Prune selection for deleted files.
    const names = new Set(this.datasets.map((d) => d.name));
    for (const s of this.selected) if (!names.has(s)) this.selected.delete(s);

    if (this.datasets.length === 0) {
      host.append(
        el("div", { class: "p-8 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500" },
          "Sin datasets locales. Descarga uno o coloca .txt/.bin en datasets/."),
      );
      this.root.append(host);
      return;
    }

    // Table header
    const head = el("div", {
      class: "grid grid-cols-[auto_1fr_auto_auto] gap-3 px-4 py-2 text-[11px] font-semibold uppercase tracking-wide text-slate-600 border-b border-slate-800/70",
    }, [
      el("span", {}, "✓"),
      el("span", {}, "Nombre"),
      el("span", {}, "Tamaño"),
      el("span", {}, "Acciones"),
    ]);

    const rows = el("div", { class: "space-y-2 mt-3" });
    for (const d of this.datasets) rows.append(this.row(d));

    const listCard = el("div", { class: "glass-card rounded-2xl p-5" }, [
      el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-3" }, "Datasets locales"),
      head,
      rows,
    ]);
    host.append(listCard);

    this.root.append(host);
    this.root.append(this.combineCard());
    this.updateCombineCount();
  }

  // ------------------------------------------------------------- row
  row(d) {
    const extBadge = d.ext === "bin"
      ? badge("bin", "violet")
      : badge("txt", "sky");

    const cb = el("input", { type: "checkbox", class: "w-4 h-4 rounded accent-violet-500 cursor-pointer" });
    if (this.selected.has(d.name)) cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) this.selected.add(d.name);
      else this.selected.delete(d.name);
      this.updateCombineCount();
      if (this._renderWeights) this._renderWeights();
    });

    const infoBtn = el("button", { class: rowBtnCls, onclick: () => this.showInfo(d) }, "info");
    const renameBtn = el("button", { class: rowBtnCls, onclick: () => this.rename(d) }, "renombrar");
    const delBtn = el("button", {
      class: rowBtnCls + " bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border-rose-500/20",
      onclick: () => this.remove(d),
    }, "borrar");

    return el("div", {
      class: "lift grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center px-4 py-2.5 rounded-xl bg-slate-950/70 border border-slate-800/70",
    }, [
      cb,
      el("span", { class: "flex items-center gap-2 text-sm text-slate-200 font-mono truncate" }, [extBadge, d.name]),
      el("span", { class: "text-xs text-slate-500 tnum font-mono" }, fmtBytes(d.size_bytes)),
      el("div", { class: "flex gap-1.5" }, [infoBtn, renameBtn, delBtn]),
    ]);
  }

  updateCombineCount() {
    const el2 = document.getElementById("combine-count");
    if (el2) el2.textContent = `${this.selected.size} seleccionados`;
  }

  // ------------------------------------------------------------- combine
  combineCard() {
    const count = el("span", { id: "combine-count", class: "text-xs text-slate-400 tnum" }, "0 seleccionados");
    const weightsHost = el("div", { class: "space-y-2 mt-3" });
    const outName = el("input", { type: "text", value: "weighted_mix_web.txt", class: inputBaseCls });
    const msg = el("p", { class: "text-xs text-slate-400 mt-3" }, "");

    // Show selected datasets with editable weights.
    const renderWeights = () => {
      clear(weightsHost);
      const sel = [...this.selected].sort();
      if (sel.length < 2) {
        weightsHost.append(
          el("p", { class: "text-xs text-slate-500" }, "Selecciona al menos 2 datasets de la tabla de arriba."),
        );
        return;
      }
      weightsHost.append(
        el("div", { class: "grid grid-cols-[1fr_90px] gap-3 text-[11px] font-semibold uppercase tracking-wide text-slate-600 px-1" }, [
          el("span", {}, "Dataset"),
          el("span", {}, "Peso"),
        ]),
      );
      for (const name of sel) {
        const w = el("input", {
          type: "number", value: 1, min: 0, step: 0.5,
          class: "w-full px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-xs tnum focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition",
        });
        weightsHost.append(
          el("div", { class: "grid grid-cols-[1fr_90px] gap-3 items-center px-1 py-1" }, [
            el("span", { class: "text-sm text-slate-200 font-mono truncate" }, name),
            w,
          ]),
        );
      }
      // Store weights for the run handler.
      weightsHost.dataset.ready = "1";
      weightsHost._inputs = weightsHost.querySelectorAll("input[type=number]");
    };

    this._renderWeights = renderWeights;

    const run = async () => {
      if (!weightsHost._inputs || weightsHost._inputs.length < 2) {
        msg.textContent = "Selecciona al menos 2 datasets en la tabla.";
        msg.className = "text-xs text-rose-400 mt-3";
        return;
      }
      const names = [...this.selected].sort();
      const sources = [];
      weightsHost._inputs.forEach((w, i) => {
        if (names[i]) sources.push({ path: names[i], weight: Number(w.value) || 0 });
      });
      msg.textContent = "Combinando…";
      msg.className = "text-xs text-violet-300 mt-3";
      try {
        const res = await api.post("/api/datasets/combine", {
          sources,
          output: outName.value.trim() || "weighted_mix_web.txt",
        });
        msg.textContent = `Mix creado: ${res.output}`;
        msg.className = "text-xs text-emerald-300 mt-3";
        toast.success(`Mix creado: ${res.output}`);
        this.selected.clear();
        this.render();
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-xs text-rose-400 mt-3";
      }
    };

    const card = el("div", { class: "datasets-combine-card glass-card rounded-2xl p-5 mt-6" }, [
      el("div", { class: "flex items-center justify-between mb-2" }, [
        el("h3", { class: "text-[11px] font-semibold uppercase tracking-wider text-slate-500" }, "Combinar con pesos"),
        count,
      ]),
      el("p", { class: "text-xs text-slate-500 mb-2" }, "Marca datasets en la tabla de arriba, ajusta el peso de cada uno y genera un mix único."),
      weightsHost,
      el("div", { class: "mt-4" }, [
        el("label", { class: "block text-xs font-medium text-slate-400" }, ["Nombre del mix", outName]),
      ]),
      el("div", { class: "mt-4" }, [
        btn.success(iconSvg("layers", "w-3.5 h-3.5") + " Combinar", run),
      ]),
      msg,
    ]);

    return card;
  }

  // ------------------------------------------------------------- actions
  async showInfo(d) {
    try {
      const info = await api.get(`/api/datasets/info/${encodeURIComponent(d.name)}`);
      const lines = [
        `Tamaño: ${fmtBytes(info.size_bytes)}`,
        `Tipo: ${info.ext}`,
        info.lines != null ? `Líneas: ${info.lines.toLocaleString("es-ES")}` : "",
      ].filter(Boolean).join("\n");
      await modal.alert({ title: info.name, message: lines });
    } catch (e) {
      toast.error(`Error: ${e.message}`);
    }
  }

  async rename(d) {
    const to = await modal.prompt({
      title: "Renombrar dataset",
      label: "Nuevo nombre",
      value: d.name,
    });
    if (!to || to === d.name) return;
    try {
      await api.post("/api/datasets/rename", { from: d.name, to });
      toast.success(`Renombrado a ${to}`);
      this.render();
    } catch (e) {
      toast.error(`Error: ${e.message}`);
    }
  }

  async remove(d) {
    const ok = await modal.confirm({
      title: "¿Borrar dataset?",
      message: d.name,
      confirmText: "Borrar",
      danger: true,
    });
    if (!ok) return;
    try {
      await api.delete(`/api/datasets/delete/${encodeURIComponent(d.name)}`);
      toast.success(`${d.name} borrado`);
      this.selected.delete(d.name);
      this.render();
    } catch (e) {
      toast.error(`Error: ${e.message}`);
    }
  }
}

const inputBaseCls =
  "w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition";
const rowBtnCls =
  "px-2 py-1 rounded-md text-[11px] font-medium bg-slate-800/70 hover:bg-slate-700/70 text-slate-300 border border-slate-700/50 transition active:scale-[0.98]";
