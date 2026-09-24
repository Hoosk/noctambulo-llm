// App bootstrap: token gate + sidebar layout + view router.

import { getToken, setToken } from "./core/api.js";
import { on } from "./core/events.js";
import { getJob, initJobStore, pollStatus, JobState } from "./core/store.js";
import * as sse from "./core/sse.js";
import { catLogoSvg, iconSvg } from "./components/ui.js";

import TrainView from "./views/train.js";
import GenerateView from "./views/generate.js";
import ExportView from "./views/export.js";
import ModelsView from "./views/models.js";
import DevicesView from "./views/devices.js";
import DatasetsView from "./views/datasets.js";

const VIEWS = {
  train: TrainView,
  generate: GenerateView,
  export: ExportView,
  models: ModelsView,
  devices: DevicesView,
  datasets: DatasetsView,
};

const NAV = [
  { name: "train", label: "Entrenar", icon: "activity" },
  { name: "datasets", label: "Datasets", icon: "database" },
  { name: "generate", label: "Generar", icon: "sparkles" },
  { name: "export", label: "Exportar", icon: "upload" },
  { name: "models", label: "Modelos", icon: "box" },
  { name: "devices", label: "GPU", icon: "cpu" },
];

const CONN_MARKUP = `
  <span class="conn-badge inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border border-slate-700 bg-slate-800/80 text-slate-400">
    <span class="conn-dot w-1.5 h-1.5 rounded-full bg-slate-500"></span>
    <span class="conn-text">conectando…</span>
  </span>`;

function renderTokenGate() {
  document.getElementById("app").innerHTML = `
    <div class="min-h-screen flex items-center justify-center p-4">
      <div class="glass-card max-w-sm w-full p-8 rounded-2xl animate-scale-in">
        <div class="flex items-center gap-4 mb-8">
          ${catLogoSvg("w-12 h-12")}
          <div class="min-w-0">
            <h1 class="text-lg font-semibold text-white tracking-tight">Kuro-Neko</h1>
            <p class="text-xs text-slate-500">Control de entrenamiento</p>
          </div>
        </div>
        <label class="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">API Token</label>
        <input id="token-input" type="password" placeholder="Pega tu token aquí"
          class="w-full px-3 py-2.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition mb-4" />
        <button id="token-submit"
          class="w-full py-2.5 rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium shadow-lg shadow-violet-600/25 transition active:scale-[0.98]">
          Conectar
        </button>
        <p id="token-error" class="text-xs text-rose-400 mt-3 hidden"></p>
      </div>
    </div>`;

  const input = document.getElementById("token-input");
  const submit = () => {
    const t = input.value.trim();
    if (!t) return;
    setToken(t);
    // Verify token quickly (cheap ping; adapter enumeration is slow).
    fetch("/api/ping", { headers: { "X-API-Token": t } })
      .then((r) => {
        if (r.ok) {
          bootstrap();
        } else {
          const err = document.getElementById("token-error");
          err.textContent = "Token rechazado. Verifícalo e inténtalo de nuevo.";
          err.classList.remove("hidden");
          setToken("");
        }
      })
      .catch(() => {
        const err = document.getElementById("token-error");
        err.textContent = "No se pudo conectar con el servidor.";
        err.classList.remove("hidden");
      });
  };
  input.addEventListener("keydown", (e) => e.key === "Enter" && submit());
  document.getElementById("token-submit").addEventListener("click", submit);
}

function navButton({ name, label, icon }) {
  return `
    <button data-view="${name}"
      class="view-btn group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition">
      ${iconSvg(icon, "w-4 h-4 shrink-0")}
      <span>${label}</span>
    </button>`;
}

function mountLayout() {
  document.getElementById("app").innerHTML = `
    <div class="min-h-screen flex">
      <div id="sidebar-backdrop" class="fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-sm hidden lg:hidden"></div>

      <aside id="sidebar"
        class="fixed inset-y-0 left-0 z-40 w-64 flex flex-col p-4 glass-card rounded-none border-y-0 border-l-0
               transition-transform duration-200 -translate-x-full lg:translate-x-0">
        <div class="flex items-center gap-3 px-1 mb-6">
          ${catLogoSvg("w-10 h-10")}
          <div class="min-w-0">
            <h1 class="text-base font-semibold text-white tracking-tight leading-tight">Kuro-Neko</h1>
            <p class="text-[11px] text-slate-500">Control de entrenamiento</p>
          </div>
        </div>

        <nav class="flex flex-col gap-1">
          ${NAV.map(navButton).join("")}
        </nav>

        <div class="flex-1"></div>

        <div class="rounded-xl border border-slate-800/70 bg-slate-950/60 p-3">
          <div id="global-progress" class="mb-3 hidden">
            <div class="flex items-center justify-between gap-2 text-[11px] mb-1.5">
              <span id="global-progress-label" class="text-slate-500 truncate">—</span>
              <span id="global-progress-pct" class="tnum font-mono text-slate-400">0%</span>
            </div>
            <div class="h-1.5 rounded-full bg-slate-800 overflow-hidden">
              <div id="global-progress-bar"
                class="h-full w-0 rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300"></div>
            </div>
          </div>
          <div class="flex items-center justify-center">${CONN_MARKUP}</div>
        </div>
      </aside>

      <div class="flex-1 min-w-0 flex flex-col lg:pl-64">
        <header class="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-slate-800/70 bg-slate-950/60 backdrop-blur sticky top-0 z-20">
          <button id="menu-btn" class="p-2 -m-2 text-slate-300 hover:text-white transition" aria-label="Abrir menú">
            ${iconSvg("menu", "w-5 h-5")}
          </button>
          ${catLogoSvg("w-7 h-7")}
          <h1 class="text-base font-semibold text-white tracking-tight">Kuro-Neko</h1>
          <span class="ml-auto">${CONN_MARKUP}</span>
        </header>

        <main id="view-root" class="flex-1 px-4 sm:px-6 lg:px-8 py-6 max-w-6xl w-full mx-auto"></main>
      </div>
    </div>`;

  // Mobile drawer wiring.
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const setDrawer = (open) => {
    sidebar.classList.toggle("translate-x-0", open);
    sidebar.classList.toggle("-translate-x-full", !open);
    backdrop.classList.toggle("hidden", !open);
  };
  document.getElementById("menu-btn").addEventListener("click", () => setDrawer(true));
  backdrop.addEventListener("click", () => setDrawer(false));
  document.querySelectorAll(".view-btn").forEach((b) =>
    b.addEventListener("click", () => {
      setDrawer(false);
      switchView(b.dataset.view);
    }),
  );
}

const viewInstances = {};

function switchView(name) {
  const root = document.getElementById("view-root");
  document.querySelectorAll(".view-btn").forEach((b) => {
    const active = b.dataset.view === name;
    b.classList.toggle("nav-glow", active);
    b.classList.toggle("text-white", active);
    b.classList.toggle("font-medium", active);
    b.classList.toggle("text-slate-400", !active);
  });

  if (!viewInstances[name]) {
    viewInstances[name] = new VIEWS[name](root);
  }
  // Replay the entrance animation on every switch.
  root.classList.remove("animate-fade-slide");
  void root.offsetWidth;
  root.classList.add("animate-fade-slide");
  viewInstances[name].render();
}

function updateBadge(connected) {
  document.querySelectorAll(".conn-badge").forEach((b) => {
    const dot = b.querySelector(".conn-dot");
    const text = b.querySelector(".conn-text");
    if (!dot || !text) return;
    if (connected) {
      b.className = "conn-badge inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/15 text-emerald-300";
      dot.className = "conn-dot w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-dot";
      text.textContent = "en línea";
    } else {
      b.className = "conn-badge inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border border-rose-500/30 bg-rose-500/15 text-rose-300";
      dot.className = "conn-dot w-1.5 h-1.5 rounded-full bg-rose-400";
      text.textContent = "desconectado";
    }
  });
}

const GLOBAL_LABELS = {
  [JobState.RUNNING]: "entrenando…",
  [JobState.DONE]: "completado",
  [JobState.STOPPED]: "detenido",
  [JobState.ERROR]: "error",
};

function updateGlobalProgress(job) {
  const wrap = document.getElementById("global-progress");
  const label = document.getElementById("global-progress-label");
  const pct = document.getElementById("global-progress-pct");
  const bar = document.getElementById("global-progress-bar");
  if (!wrap || !label || !pct || !bar) return;

  const running = job.status === JobState.RUNNING;
  const hasJob = running || GLOBAL_LABELS[job.status];
  wrap.classList.toggle("hidden", !hasJob);

  label.textContent = running
    ? (job.paused ? "pausado" : job.stopping ? "deteniendo…" : GLOBAL_LABELS[JobState.RUNNING])
    : (GLOBAL_LABELS[job.status] || "—");
  const value = Number.isFinite(job.pct) ? job.pct : 0;
  pct.textContent = `${value.toFixed(0)}%`;
  bar.style.width = `${value}%`;
  bar.classList.toggle("progress-shimmer", running && !job.paused && !job.stopping);
}

function bootstrap() {
  mountLayout();
  initJobStore();

  // Register BEFORE connecting so the initial status is not missed.
  on("sse:status", ({ connected }) => updateBadge(connected));
  on("job:changed", updateGlobalProgress);
  updateGlobalProgress(getJob());

  // Background heartbeat: confirm the API is reachable regardless of SSE state.
  setInterval(() => {
    fetch("/api/ping", { headers: { "X-API-Token": getToken() } })
      .then((r) => updateBadge(r.ok))
      .catch(() => updateBadge(false));
  }, 10000);

  // Polling fallback so metrics update even if the SSE stream is dropped.
  setInterval(pollStatus, 2000);

  sse.connect();
  switchView("train");
}

document.addEventListener("DOMContentLoaded", () => {
  // A 401 anywhere (token revoked/expired) drops back to the gate.
  window.addEventListener("kuro:unauthorized", () => {
    sse.disconnect();
    renderTokenGate();
  });

  if (getToken()) {
    // Validate token; fall back to gate if it fails.
    fetch("/api/ping", { headers: { "X-API-Token": getToken() } })
      .then((r) => (r.ok ? bootstrap() : renderTokenGate()))
      .catch(() => renderTokenGate());
  } else {
    renderTokenGate();
  }
});
