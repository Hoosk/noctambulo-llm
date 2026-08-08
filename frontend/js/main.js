// App bootstrap: token gate + view router.

import { getToken, setToken } from "./core/api.js";
import { on } from "./core/events.js";
import { initJobStore } from "./core/store.js";
import * as sse from "./core/sse.js";

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

function renderTokenGate() {
  document.getElementById("app").innerHTML = `
    <div class="min-h-screen flex items-center justify-center bg-slate-950">
      <div class="max-w-sm w-full p-8 rounded-2xl bg-slate-900 border border-slate-800 shadow-xl">
        <div class="flex items-center gap-3 mb-6">
          <span class="text-3xl">🐈‍⬛</span>
          <div>
            <h1 class="text-lg font-semibold text-white">Kuro-Neko</h1>
            <p class="text-xs text-slate-400">Control de entrenamiento</p>
          </div>
        </div>
        <label class="block text-sm text-slate-300 mb-2">API Token</label>
        <input id="token-input" type="password" placeholder="Pega tu token aquí"
          class="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500 mb-4" />
        <button id="token-submit"
          class="w-full py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium transition">
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
    // Verify token quickly.
    fetch("/api/devices", { headers: { "X-API-Token": t } })
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

function mountLayout() {
  document.getElementById("app").innerHTML = `
    <div class="min-h-screen bg-slate-950 text-slate-200 flex flex-col">
      <header class="flex items-center gap-4 px-6 py-4 border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-20">
        <span class="text-2xl">🐈‍⬛</span>
        <h1 class="text-lg font-semibold text-white tracking-tight">Kuro-Neko</h1>
        <span id="conn-badge" class="ml-1 text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">conectando…</span>
        <nav class="ml-auto flex gap-1">
          <button data-view="train" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Entrenar</button>
          <button data-view="datasets" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Datasets</button>
          <button data-view="generate" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Generar</button>
          <button data-view="export" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Exportar</button>
          <button data-view="models" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">Modelos</button>
          <button data-view="devices" class="view-btn px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800 transition">GPU</button>
        </nav>
      </header>
      <main id="view-root" class="flex-1 px-6 py-6 max-w-6xl w-full mx-auto"></main>
    </div>`;
}

const viewInstances = {};

function switchView(name) {
  const root = document.getElementById("view-root");
  document.querySelectorAll(".view-btn").forEach((b) => {
    const active = b.dataset.view === name;
    b.classList.toggle("text-white", active);
    b.classList.toggle("bg-slate-800", active);
    b.classList.toggle("text-slate-400", !active);
  });

  if (!viewInstances[name]) {
    viewInstances[name] = new VIEWS[name](root);
  }
  viewInstances[name].render();
}

function updateBadge(connected) {
  const badge = document.getElementById("conn-badge");
  if (!badge) return;
  if (connected) {
    badge.textContent = "en línea";
    badge.className = "ml-1 text-xs px-2 py-0.5 rounded-full bg-emerald-900 text-emerald-300";
  } else {
    badge.textContent = "desconectado";
    badge.className = "ml-1 text-xs px-2 py-0.5 rounded-full bg-rose-900 text-rose-300";
  }
}

function bootstrap() {
  mountLayout();
  initJobStore();

  // Register BEFORE connecting so the initial status is not missed.
  on("sse:status", ({ connected }) => updateBadge(connected));

  document.querySelectorAll(".view-btn").forEach((b) => {
    b.addEventListener("click", () => switchView(b.dataset.view));
  });

  // Background heartbeat: confirm the API is reachable regardless of SSE state.
  setInterval(() => {
    fetch("/api/devices", { headers: { "X-API-Token": getToken() } })
      .then((r) => updateBadge(r.ok))
      .catch(() => updateBadge(false));
  }, 10000);

  sse.connect();
  switchView("train");
}

document.addEventListener("DOMContentLoaded", () => {
  if (getToken()) {
    // Validate token; fall back to gate if it fails.
    fetch("/api/devices", { headers: { "X-API-Token": getToken() } })
      .then((r) => (r.ok ? bootstrap() : renderTokenGate()))
      .catch(() => renderTokenGate());
  } else {
    renderTokenGate();
  }
});
