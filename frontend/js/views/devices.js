// Devices view: list WGPU adapters with real specs.

import { api } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";

export default class DevicesView {
  constructor(root) {
    this.root = root;
  }

  async render() {
    clear(this.root);
    this.root.append(
      el("div", { class: "mb-6" }, [
        el("h2", { class: "text-xl font-semibold text-white" }, "Dispositivos GPU"),
        el("p", { class: "text-sm text-slate-400 mt-1" }, "Adaptadores WGPU detectados en el host del servidor."),
      ]),
    );

    let devices = [];
    try {
      devices = await api.get("/api/devices");
    } catch (e) {
      this.root.append(el("p", { class: "text-sm text-rose-400" }, `Error: ${e.message}`));
      return;
    }

    if (devices.length === 0) {
      this.root.append(
        el("div", { class: "p-10 rounded-2xl border border-dashed border-slate-700 text-center text-slate-500" },
          "No se detectaron adaptadores."),
      );
      return;
    }

    const list = el("ul", { class: "grid grid-cols-1 md:grid-cols-2 gap-4" });
    for (const d of devices) {
      list.append(this.card(d));
    }
    this.root.append(list);
  }

  card(d) {
    const typeBadge = {
      "Discrete GPU": ["bg-violet-900 text-violet-300", "⚡"],
      "Integrated GPU": ["bg-sky-900 text-sky-300", "🔧"],
      "CPU": ["bg-slate-800 text-slate-300", "🧠"],
      "Auto": ["bg-emerald-900 text-emerald-300", "✨"],
    }[d.device_type] || ["bg-slate-800 text-slate-300", "❓"];
    const [cls, icon] = typeBadge;

    const row = (label, value) =>
      el("div", { class: "flex justify-between gap-4 text-xs py-1" }, [
        el("span", { class: "text-slate-500" }, label),
        el("span", { class: "text-slate-300 font-mono text-right" }, value),
      ]);

    return el("li", { class: "rounded-2xl border border-slate-800 bg-slate-900 p-5" }, [
      el("div", { class: "flex items-center justify-between mb-3" }, [
        el("span", { class: "text-sm font-medium text-white flex items-center gap-2" }, [
          el("span", {}, icon),
          d.name,
        ]),
        el("span", { class: `text-xs px-2 py-0.5 rounded-full ${cls}` }, d.device_type),
      ]),
      el("div", { class: "border-t border-slate-800 pt-2" }, [
        row("Backend", d.backend),
        row("Driver", d.driver || "—"),
        row("Vendor ID", d.vendor ? `0x${d.vendor.toString(16)}` : "—"),
        row("Device ID", d.device ? `0x${d.device.toString(16)}` : "—"),
        row("VRAM aprox.", d.vram_bytes ? fmtBytes(d.vram_bytes) : "—"),
      ]),
    ]);
  }
}
