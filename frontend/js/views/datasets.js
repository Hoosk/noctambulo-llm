// Datasets view: download from HF, organize local corpus, combine with weights.

import { api } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";

export default class DatasetsView {
  constructor(root) {
    this.root = root;
    this.datasets = [];
    this.selected = new Set();
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("div", { class: "flex items-end justify-between mb-6" }, [
        el("div", {}, [
          el("h2", { class: "text-xl font-semibold text-white" }, "Datasets"),
          el("p", { class: "text-sm text-slate-400 mt-1" }, "Descarga, organiza y combina corpus."),
        ]),
        el("button", {
          class: "px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm text-slate-200 transition",
          onclick: () => this.render(),
        }, "Refrescar"),
      ]),
    );

    this.root.append(this.downloadCard());
    this.root.append(el("div", { class: "h-6" }));
    this.root.append(this.searchCard());

    await this.loadList();
  }

  // ------------------------------------------------------------- search HF
  searchCard() {
    const searchInput = el("input", {
      type: "text",
      placeholder: "Buscar datasets en Hugging Face (ej. cosmopedia, stories, code)...",
      class: inputCls,
    });
    const msg = el("p", { class: "text-sm text-slate-400 mt-3" }, "");
    const results = el("div", { class: "mt-3 space-y-2" });

    const search = async () => {
      const q = searchInput.value.trim();
      if (!q) return;
      clear(results);
      msg.textContent = "Buscando…";
      msg.className = "text-sm text-violet-300 mt-3";
      try {
        const hits = await api.get(`/api/datasets/search?q=${encodeURIComponent(q)}`);
        msg.textContent = "";
        if (hits.length === 0) {
          results.append(el("p", { class: "text-sm text-slate-500" }, "Sin resultados."));
          return;
        }
        for (const h of hits) {
          results.append(this.searchRow(h));
        }
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-sm text-rose-400 mt-3";
      }
    };

    searchInput.addEventListener("keydown", (e) => e.key === "Enter" && search());

    return el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("div", { class: "flex items-center justify-between mb-2" }, [
        el("h3", { class: "text-sm font-medium text-slate-300" }, "Buscar en Hugging Face"),
      ]),
      el("div", { class: "flex gap-3" }, [
        el("div", { class: "flex-1" }, [searchInput]),
        el("button", {
          class: "px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium transition self-start",
          onclick: search,
        }, "🔍 Buscar"),
      ]),
      msg,
      results,
    ]);
  }

  searchRow(h) {
    const lang = h.tags && h.tags.length
      ? el("span", { class: "text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300" }, h.tags[0].replace("language:", ""))
      : null;

    const dl = async () => {
      try {
        await api.post("/api/datasets/download", { repoId: h.id });
        this.loadList();
      } catch (e) {
        alert(`Error descargando: ${e.message}`);
      }
    };

    return el("div", {
      class: "rounded-xl bg-slate-950 border border-slate-800 px-4 py-3 flex items-center gap-3",
    }, [
      el("div", { class: "flex-1 min-w-0" }, [
        el("div", { class: "flex items-center gap-2" }, [
          el("span", { class: "text-sm text-slate-200 font-mono truncate" }, h.id),
          lang,
        ]),
        el("div", { class: "text-xs text-slate-500 mt-0.5" }, [
          `${h.downloads.toLocaleString("es-ES")} descargas · ${h.likes.toLocaleString("es-ES")} likes`,
        ]),
      ]),
      el("button", {
        class: "px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium transition shrink-0",
        onclick: dl,
      }, "⬇ Descargar"),
    ]);
  }

  // ------------------------------------------------------------- download
  downloadCard() {
    const repoInput = el("input", {
      type: "text",
      placeholder: "HuggingFaceTB/cosmopedia-100k",
      class: inputCls,
    });
    const fileInput = el("input", {
      type: "text",
      placeholder: "archivo dentro del repo (opcional)",
      class: inputCls,
    });
    const msg = el("p", { class: "text-sm text-slate-400 mt-3" }, "");

    const download = async () => {
      const repoId = repoInput.value.trim();
      if (!repoId) {
        msg.textContent = "Indica un repo de Hugging Face.";
        msg.className = "text-sm text-rose-400 mt-3";
        return;
      }
      msg.textContent = "Descargando… (puede tardar)";
      msg.className = "text-sm text-violet-300 mt-3";
      try {
        const res = await api.post("/api/datasets/download", {
          repoId,
          filename: fileInput.value.trim() || "TinyStories-train.txt",
        });
        msg.textContent = `Descargado: ${res.name}`;
        msg.className = "text-sm text-emerald-300 mt-3";
        this.loadList();
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-sm text-rose-400 mt-3";
      }
    };

    return el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("h3", { class: "text-sm font-medium text-slate-300 mb-4" }, "Descargar de Hugging Face"),
      el("div", { class: "grid grid-cols-1 sm:grid-cols-2 gap-3" }, [
        el("label", { class: "block text-xs text-slate-400" }, ["Repo ID", repoInput]),
        el("label", { class: "block text-xs text-slate-400" }, ["Archivo (opcional)", fileInput]),
      ]),
      el("button", {
        class: "mt-4 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition",
        onclick: download,
      }, "⬇ Descargar"),
      msg,
    ]);
  }

  // ------------------------------------------------------------- listing
  async loadList() {
    const host = document.createElement("div");
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
        el("div", { class: "p-8 rounded-2xl border border-dashed border-slate-700 text-center text-slate-500" },
          "Sin datasets locales. Descarga uno o coloca .txt/.bin en datasets/."),
      );
      this.root.append(host);
      return;
    }

    // Table header
    const head = el("div", {
      class: "grid grid-cols-[auto_1fr_auto_auto] gap-3 px-4 py-2 text-xs text-slate-500 border-b border-slate-800",
    }, [
      el("span", {}, "✓"),
      el("span", {}, "Nombre"),
      el("span", {}, "Tamaño"),
      el("span", {}, "Acciones"),
    ]);

    const rows = el("div", { class: "space-y-2 mt-2" });
    for (const d of this.datasets) rows.append(this.row(d));

    const listCard = el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("h3", { class: "text-sm font-medium text-slate-300 mb-3" }, "Datasets locales"),
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
      ? el("span", { class: "text-[10px] px-1.5 py-0.5 rounded bg-violet-900 text-violet-200 font-mono" }, "bin")
      : el("span", { class: "text-[10px] px-1.5 py-0.5 rounded bg-sky-900 text-sky-200 font-mono" }, "txt");

    const cb = el("input", { type: "checkbox", class: "accent-violet-500" });
    if (this.selected.has(d.name)) cb.checked = true;
    cb.addEventListener("change", () => {
      if (cb.checked) this.selected.add(d.name);
      else this.selected.delete(d.name);
      this.updateCombineCount();
      if (this._renderWeights) this._renderWeights();
    });

    const infoBtn = el("button", { class: "px-2 py-1 rounded text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition", onclick: () => this.showInfo(d) }, "info");
    const renameBtn = el("button", { class: "px-2 py-1 rounded text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition", onclick: () => this.rename(d) }, "renombrar");
    const delBtn = el("button", {
      class: "px-2 py-1 rounded text-xs bg-rose-900 hover:bg-rose-800 text-rose-200 transition",
      onclick: () => this.remove(d),
    }, "borrar");

    return el("div", {
      class: "grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800",
    }, [
      cb,
      el("span", { class: "flex items-center gap-2 text-sm text-slate-200 font-mono truncate" }, [extBadge, d.name]),
      el("span", { class: "text-xs text-slate-500" }, fmtBytes(d.size_bytes)),
      el("div", { class: "flex gap-2" }, [infoBtn, renameBtn, delBtn]),
    ]);
  }

  updateCombineCount() {
    const el2 = document.getElementById("combine-count");
    if (el2) el2.textContent = `${this.selected.size} seleccionados`;
  }

  // ------------------------------------------------------------- combine
  combineCard() {
    const count = el("span", { id: "combine-count", class: "text-xs text-slate-400" }, "0 seleccionados");
    const weightsHost = el("div", { class: "space-y-2 mt-3" });
    const outName = el("input", { type: "text", value: "weighted_mix_web.txt", class: inputCls });
    const msg = el("p", { class: "text-sm text-slate-400 mt-3" }, "");

    // Show selected datasets with editable weights.
    const renderWeights = () => {
      clear(weightsHost);
      const sel = [...this.selected].sort();
      if (sel.length < 2) {
        weightsHost.append(
          el("p", { class: "text-sm text-slate-500" }, "Selecciona al menos 2 datasets de la tabla de arriba."),
        );
        return;
      }
      weightsHost.append(
        el("div", { class: "grid grid-cols-[1fr_90px] gap-3 text-xs text-slate-500 px-1" }, [
          el("span", {}, "Dataset"),
          el("span", {}, "Peso"),
        ]),
      );
      for (const name of sel) {
        const w = el("input", {
          type: "number", value: 1, min: 0, step: 0.5,
          class: "w-full px-2 py-1 rounded bg-slate-950 border border-slate-700 text-white text-xs",
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
        msg.className = "text-sm text-rose-400 mt-3";
        return;
      }
      const names = [...this.selected].sort();
      const sources = [];
      weightsHost._inputs.forEach((w, i) => {
        if (names[i]) sources.push({ path: names[i], weight: Number(w.value) || 0 });
      });
      msg.textContent = "Combinando…";
      msg.className = "text-sm text-violet-300 mt-3";
      try {
        const res = await api.post("/api/datasets/combine", {
          sources,
          output: outName.value.trim() || "weighted_mix_web.txt",
        });
        msg.textContent = `Mix creado: ${res.output}`;
        msg.className = "text-sm text-emerald-300 mt-3";
        this.selected.clear();
        this.render();
      } catch (e) {
        msg.textContent = `Error: ${e.message}`;
        msg.className = "text-sm text-rose-400 mt-3";
      }
    };

    const card = el("div", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5 mt-6" }, [
      el("div", { class: "flex items-center justify-between mb-2" }, [
        el("h3", { class: "text-sm font-medium text-slate-300" }, "Combinar con pesos"),
        count,
      ]),
      el("p", { class: "text-xs text-slate-500 mb-2" }, "Marca datasets en la tabla de arriba, ajusta el peso de cada uno y genera un mix único."),
      weightsHost,
      el("div", { class: "mt-4" }, [
        el("label", { class: "block text-xs text-slate-400" }, ["Nombre del mix", outName]),
      ]),
      el("button", {
        class: "mt-4 px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white text-sm font-medium transition",
        onclick: run,
      }, "⚗ Combinar"),
      msg,
    ]);

    return card;
  }

  // ------------------------------------------------------------- actions
  async showInfo(d) {
    try {
      const info = await api.get(`/api/datasets/info/${encodeURIComponent(d.name)}`);
      alert(
        `Dataset: ${info.name}\n` +
        `Tamaño: ${fmtBytes(info.size_bytes)}\n` +
        `Tipo: ${info.ext}\n` +
        (info.lines != null ? `Líneas: ${info.lines}\n` : "")
      );
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  }

  async rename(d) {
    const to = prompt("Nuevo nombre:", d.name);
    if (!to || to === d.name) return;
    try {
      await api.post("/api/datasets/rename", { from: d.name, to });
      this.render();
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  }

  async remove(d) {
    if (!confirm(`¿Borrar ${d.name}?`)) return;
    try {
      await api.delete(`/api/datasets/delete/${encodeURIComponent(d.name)}`);
      this.selected.delete(d.name);
      this.render();
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  }
}

const inputCls =
  "w-full px-3 py-2 mt-1 rounded-lg bg-slate-950 border border-slate-700 text-white focus:outline-none focus:ring-2 focus:ring-violet-500";
