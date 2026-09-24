// Devices view: list WGPU adapters with real specs.

import { api } from "../core/api.js";
import { clear, el, fmtBytes } from "../core/dom.js";
import { viewTitle, badge, icon, stagger } from "../components/ui.js";

const TYPE_INFO = {
  "Discrete GPU": { tone: "violet", icon: "zap" },
  "Integrated GPU": { tone: "sky", icon: "cpu" },
  "CPU": { tone: "slate", icon: "cpu" },
  "Auto": { tone: "emerald", icon: "sparkles" },
};

export default class DevicesView {
  constructor(root) {
    this.root = root;
  }

  async render() {
    clear(this.root);
    this.root.append(
      viewTitle("Dispositivos GPU", "Adaptadores WGPU detectados en el host del servidor."),
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
        el("div", { class: "p-10 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500" },
          "No se detectaron adaptadores."),
      );
      return;
    }

    const list = el("ul", { class: "grid grid-cols-1 md:grid-cols-2 gap-4" });
    for (const d of devices) {
      list.append(this.card(d));
    }
    this.root.append(list);
    stagger(this.root);
  }

  card(d) {
    const info = TYPE_INFO[d.device_type] || { tone: "slate", icon: "info" };

    const chipCls = {
      violet: "bg-violet-500/15 text-violet-300",
      sky: "bg-sky-500/15 text-sky-300",
      slate: "bg-slate-700/40 text-slate-300",
      emerald: "bg-emerald-500/15 text-emerald-300",
    }[info.tone];

    const row = (label, value) =>
      el("div", { class: "flex justify-between gap-4 text-xs py-1.5" }, [
        el("span", { class: "text-slate-500" }, label),
        el("span", { class: "text-slate-300 font-mono text-right tnum" }, value),
      ]);

    return el("li", { class: "lift glass-card rounded-2xl p-5" }, [
      el("div", { class: "flex items-center justify-between gap-3 mb-3" }, [
        el("span", { class: "text-sm font-medium text-white flex items-center gap-2.5 min-w-0" }, [
          el("span", { class: `w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${chipCls}` }, [
            icon(info.icon, "w-4 h-4"),
          ]),
          el("span", { class: "truncate" }, d.name),
        ]),
        badge(d.device_type, info.tone),
      ]),
      el("div", { class: "border-t border-slate-800/70 pt-2" }, [
        row("Backend", d.backend),
        row("Driver", d.driver || "—"),
        row("Vendor ID", d.vendor ? `0x${d.vendor.toString(16)}` : "—"),
        row("Device ID", d.device ? `0x${d.device.toString(16)}` : "—"),
        row("VRAM aprox.", d.vram_bytes ? fmtBytes(d.vram_bytes) : "—"),
      ]),
    ]);
  }
}
