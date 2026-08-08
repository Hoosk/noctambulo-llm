// Devices view: list WGPU devices available on the host.

import { api } from "../core/api.js";
import { clear, el } from "../core/dom.js";

export default class DevicesView {
  constructor(root) {
    this.root = root;
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("div", { class: "mb-6" }, [
        el("h2", { class: "text-xl font-semibold text-white" }, "Dispositivos GPU"),
        el("p", { class: "text-sm text-slate-400 mt-1" }, "Selectores WGPU disponibles en el host del servidor."),
      ]),
    );

    let devices = [];
    try {
      devices = await api.get("/api/devices");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    const list = el("ul", { class: "grid grid-cols-1 md:grid-cols-2 gap-3" });
    for (const d of devices) {
      const active = d.name === "DefaultDevice";
      list.append(
        el("li", {
          class: "rounded-xl border border-slate-800 bg-slate-900 px-4 py-3 flex items-center justify-between",
        }, [
          el("span", { class: "text-sm text-slate-200 font-mono" }, d.name),
          active
            ? el("span", { class: "text-xs px-2 py-0.5 rounded-full bg-emerald-900 text-emerald-300" }, "por defecto")
            : null,
        ]),
      );
    }
    this.root.append(list);
  }
}
