// Shared UI primitives: icons, cards, badges, toasts, modals, loaders.

import { el } from "../core/dom.js";

// ---------------------------------------------------------------- icons
// Minimal stroke icon set (24x24, currentColor).
const ICONS = {
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  pause: '<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  refresh: '<path d="M21 12a9 9 0 1 1-2.64-6.36"/><polyline points="21 3 21 8 16 8"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="3" x2="12" y2="15"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  search: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
  cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><line x1="9" y1="2" x2="9" y2="5"/><line x1="15" y1="2" x2="15" y2="5"/><line x1="9" y1="19" x2="9" y2="22"/><line x1="15" y1="19" x2="15" y2="22"/><line x1="2" y1="9" x2="5" y2="9"/><line x1="2" y1="15" x2="5" y2="15"/><line x1="19" y1="9" x2="22" y2="9"/><line x1="19" y1="15" x2="22" y2="15"/>',
  database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
  sparkles: '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"/>',
  box: '<path d="M21 8v8a2 2 0 0 1-1 1.73l-7 4a2 2 0 0 1-2 0l-7-4A2 2 0 0 1 3 16V8a2 2 0 0 1 1-1.73l7-4a2 2 0 0 1 2 0l7 4A2 2 0 0 1 21 8z"/><polyline points="3.3 7 12 12 20.7 7"/><line x1="12" y1="22" x2="12" y2="12"/>',
  trash: '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/>',
  pencil: '<path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>',
  info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
  zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
  activity: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
  gauge: '<path d="M12 15l3.5-3.5"/><path d="M20.3 18a10 10 0 1 0-16.6 0"/>',
  layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
  menu: '<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/>',
  loader: '<line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>',
};

export function iconSvg(name, cls = "w-4 h-4") {
  const path = ICONS[name] || ICONS.info;
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="${cls}">${path}</svg>`;
}

export function icon(name, cls = "w-4 h-4") {
  return el("span", {
    class: `inline-flex shrink-0 items-center justify-center ${cls}`,
    html: iconSvg(name, "w-full h-full"),
  });
}

// ------------------------------------------------------------------ logo
const CAT_SVG = `<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="neko-grad" x1="8" y1="6" x2="56" y2="58" gradientUnits="userSpaceOnUse">
          <stop stop-color="#a78bfa"/>
          <stop offset="0.55" stop-color="#8b5cf6"/>
          <stop offset="1" stop-color="#d946ef"/>
        </linearGradient>
      </defs>
      <path d="M11 7 L16 25 C12 30 10 34 10 39 C10 50 20 57 32 57 C44 57 54 50 54 39 C54 34 52 30 48 25 L53 7 L39 16 C35 14 29 14 25 16 L11 7 Z"
        fill="rgba(15,23,42,0.9)" stroke="url(#neko-grad)" stroke-width="2.5" stroke-linejoin="round"/>
      <path d="M21 38 Q25 42 29 38" stroke="#a78bfa" stroke-width="3" stroke-linecap="round"/>
      <path d="M35 38 Q39 42 43 38" stroke="#a78bfa" stroke-width="3" stroke-linecap="round"/>
      <path d="M30 45 Q32 47.5 34 45" stroke="#d946ef" stroke-width="2.5" stroke-linecap="round"/>
    </svg>`;

// The logo embeds a <linearGradient id>, so each insertion needs a unique id
// to keep the document valid when the logo appears more than once.
let catLogoSeq = 0;
export function catLogoSvg(cls = "w-10 h-10") {
  const uid = `neko-grad-${++catLogoSeq}`;
  return CAT_SVG.replaceAll("neko-grad", uid).replace("<svg ", `<svg class="${cls}" `);
}

// -------------------------------------------------------------- title bar
export function viewTitle(title, subtitle, actions = []) {
  return el("div", { class: "flex items-end justify-between gap-4 mb-6" }, [
    el("div", { class: "min-w-0" }, [
      el("h2", { class: "text-xl font-semibold text-white tracking-tight" }, title),
      subtitle ? el("p", { class: "text-sm text-slate-400 mt-1" }, subtitle) : null,
    ]),
    actions.length ? el("div", { class: "flex gap-2 shrink-0" }, actions) : null,
  ]);
}

const TONES = {
  slate: "bg-slate-700/40 text-slate-300",
  violet: "bg-violet-500/15 text-violet-300",
  fuchsia: "bg-fuchsia-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
};

export function statCard(label, value, opts = {}) {
  const chip = opts.icon
    ? el("div", { class: `w-8 h-8 rounded-lg flex items-center justify-center ${TONES[opts.tone] || TONES.slate}` }, [
        icon(opts.icon, "w-4 h-4"),
      ])
    : null;
  return el("div", {
    class: `rounded-xl border px-3.5 py-3 min-w-0 transition-colors ${
      opts.tone && opts.tone !== "slate"
        ? "border-slate-800 bg-slate-950/70 hover:border-slate-700"
        : "border-slate-800 bg-slate-950/70"
    }`,
  }, [
    el("div", { class: "flex items-center justify-between gap-2" }, [
      el("span", { class: "text-[11px] font-medium uppercase tracking-wide text-slate-500 truncate" }, label),
      chip,
    ]),
    el("div", { class: `tnum font-mono text-lg font-semibold mt-1 truncate ${opts.valueCls || "text-white"}` }, value),
    opts.sub ? el("div", { class: "text-[11px] text-slate-500 mt-0.5 truncate" }, opts.sub) : null,
  ]);
}

const BORDERS = {
  slate: "border-slate-700",
  violet: "border-violet-500/30",
  fuchsia: "border-fuchsia-500/30",
  emerald: "border-emerald-500/30",
  amber: "border-amber-500/30",
  rose: "border-rose-500/30",
  sky: "border-sky-500/30",
};

export function badge(text, tone = "slate", opts = {}) {
  const dot = opts.pulse
    ? el("span", { class: "relative flex w-1.5 h-1.5" }, [
        el("span", { class: "absolute inline-flex w-full h-full rounded-full opacity-60 animate-ping bg-current" }),
        el("span", { class: "relative inline-flex w-1.5 h-1.5 rounded-full bg-current" }),
      ])
    : null;
  return el("span", {
    class: `inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full border ${TONES[tone] || TONES.slate} ${BORDERS[tone] || BORDERS.slate}`,
  }, [dot, text]);
}

// Apply an entrance animation to direct children, staggered.
export function stagger(container, delay = 45) {
  if (!container) return;
  [...container.children].forEach((child, i) => {
    child.style.animation = "none";
    void child.offsetWidth; // reflow so re-renders replay the animation
    child.style.animation = `fade-slide 0.4s ease-out both`;
    child.style.animationDelay = `${i * delay}ms`;
  });
}

// ---------------------------------------------------------------- buttons
// `label` is rendered as innerHTML so callers can mix raw SVG icons + text.
const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none";

export const btn = {
  primary: (label, onclick, extra = "") =>
    el("button", {
      class: `${BTN_BASE} text-white bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 shadow-lg shadow-violet-600/25 ${extra}`,
      html: label,
      onclick,
    }),
  success: (label, onclick, extra = "") =>
    el("button", {
      class: `${BTN_BASE} text-white bg-emerald-600 hover:bg-emerald-500 shadow-lg shadow-emerald-600/20 ${extra}`,
      html: label,
      onclick,
    }),
  danger: (label, onclick, extra = "") =>
    el("button", {
      class: `${BTN_BASE} text-white bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 shadow-lg shadow-rose-600/20 ${extra}`,
      html: label,
      onclick,
    }),
  warn: (label, onclick, extra = "") =>
    el("button", {
      class: `${BTN_BASE} text-white bg-amber-600 hover:bg-amber-500 shadow-lg shadow-amber-600/20 ${extra}`,
      html: label,
      onclick,
    }),
  ghost: (label, onclick, extra = "") =>
    el("button", {
      class: `${BTN_BASE} text-slate-300 bg-slate-800/70 hover:bg-slate-700/70 border border-slate-700/60 ${extra}`,
      html: label,
      onclick,
    }),
};

// ----------------------------------------------------------------- toasts
let toastHost = null;

function ensureToastHost() {
  if (toastHost && document.body.contains(toastHost)) return toastHost;
  toastHost = el("div", {
    id: "toast-host",
    class: "fixed bottom-4 right-4 z-[70] flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)]",
  });
  document.body.append(toastHost);
  return toastHost;
}

function showToast(message, tone, iconName) {
  const host = ensureToastHost();
  const node = el("div", { class: "glass-card rounded-xl p-3.5 flex items-start gap-3 animate-fade-slide" }, [
    el("div", { class: `shrink-0 w-7 h-7 rounded-lg flex items-center justify-center ${TONES[tone] || TONES.slate}` }, [
      icon(iconName, "w-3.5 h-3.5"),
    ]),
    el("p", { class: "text-sm text-slate-200 flex-1 min-w-0 break-words" }, message),
    el("button", {
      class: "text-slate-500 hover:text-white transition shrink-0",
      onclick: () => dismiss(),
    }, icon("x", "w-3.5 h-3.5")),
  ]);

  const dismiss = () => {
    node.style.transition = "opacity .25s ease, transform .25s ease";
    node.style.opacity = "0";
    node.style.transform = "translateX(12px)";
    setTimeout(() => node.remove(), 260);
  };

  host.append(node);
  setTimeout(dismiss, 4200);
}

export const toast = {
  success: (m) => showToast(m, "emerald", "check"),
  error: (m) => showToast(m, "rose", "alert"),
  info: (m) => showToast(m, "violet", "info"),
};

// ----------------------------------------------------------------- modals
function openModal(content, { onClose } = {}) {
  const overlay = el("div", {
    class: "fixed inset-0 z-[60] flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-sm animate-fade-in",
  });
  const panel = el("div", { class: "glass-card rounded-2xl p-5 w-full max-w-sm animate-scale-in" }, content);
  overlay.append(panel);
  document.body.append(overlay);

  let closed = false;
  const close = (result) => {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", escHandler);
    overlay.remove();
    onClose?.(result);
  };
  const escHandler = (e) => e.key === "Escape" && close(null);
  document.addEventListener("keydown", escHandler);
  overlay.addEventListener("mousedown", (e) => e.target === overlay && close(null));

  return { close, overlay, panel };
}

export const modal = {
  confirm({ title = "¿Confirmar?", message = "", confirmText = "Aceptar", danger = false } = {}) {
    return new Promise((resolve) => {
      let ctl;
      const finish = (v) => { resolve(v); ctl.close(); };
      const cancelBtn = btn.ghost("Cancelar", () => finish(false));
      const okBtn = danger ? btn.danger(confirmText, () => finish(true)) : btn.primary(confirmText, () => finish(true));
      ctl = openModal([
        el("div", { class: "flex items-start gap-3 mb-4" }, [
          el("div", {
            class: `w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${danger ? TONES.rose : TONES.violet}`,
          }, [icon(danger ? "trash" : "info", "w-4 h-4")]),
          el("div", { class: "min-w-0" }, [
            el("h4", { class: "text-sm font-semibold text-white" }, title),
            message ? el("p", { class: "text-sm text-slate-400 mt-1 break-words" }, message) : null,
          ]),
        ]),
        el("div", { class: "flex gap-2 justify-end" }, [cancelBtn, okBtn]),
      ], { onClose: () => resolve(false) });
      okBtn.focus();
    });
  },

  prompt({ title = "Entrada", label = "", value = "", placeholder = "" } = {}) {
    return new Promise((resolve) => {
      let ctl;
      const input = el("input", {
        type: "text",
        value,
        placeholder,
        class: "w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition",
      });
      const finish = (v) => { resolve(v); ctl.close(); };
      const okBtn = btn.primary("Guardar", () => finish(input.value.trim() || null));
      ctl = openModal([
        el("h4", { class: "text-sm font-semibold text-white mb-3" }, title),
        label ? el("label", { class: "block text-xs text-slate-400 mb-1.5" }, label) : null,
        input,
        el("div", { class: "flex gap-2 justify-end mt-4" }, [
          btn.ghost("Cancelar", () => finish(null)),
          okBtn,
        ]),
      ], { onClose: () => resolve(null) });
      input.focus();
      input.select();
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") finish(input.value.trim() || null);
      });
    });
  },

  alert({ title = "Información", message = "" } = {}) {
    return new Promise((resolve) => {
      openModal([
        el("div", { class: "flex items-start gap-3 mb-4" }, [
          el("div", { class: "w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-violet-500/15 text-violet-300" }, [
            icon("info", "w-4 h-4"),
          ]),
          el("div", { class: "min-w-0" }, [
            el("h4", { class: "text-sm font-semibold text-white" }, title),
            message ? el("p", { class: "text-sm text-slate-400 mt-1 whitespace-pre-wrap break-words" }, message) : null,
          ]),
        ]),
        el("div", { class: "flex justify-end" }, [
          btn.primary("Entendido", () => resolve(undefined)),
        ]),
      ], { onClose: () => resolve(undefined) });
    });
  },
};
