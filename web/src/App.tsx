import { lazy, Suspense, useEffect, useState } from "react";
import type { ComponentType } from "react";
import { CatLogo, Icon, type IconName } from "./components/Icon";
import { useAppState } from "./state/appContext";
import { AppStateProvider } from "./state/AppStateProvider";
import { ModalProvider } from "./state/ModalProvider";
import { opPercent } from "./state/opsModel";
import { ToastProvider } from "./state/ToastProvider";

const TrainView = lazy(() => import("./views/TrainView").then((m) => ({ default: m.TrainView })));
const DatasetsView = lazy(() => import("./views/DatasetsView").then((m) => ({ default: m.DatasetsView })));
const GenerateView = lazy(() => import("./views/GenerateView").then((m) => ({ default: m.GenerateView })));
const ExportView = lazy(() => import("./views/ExportView").then((m) => ({ default: m.ExportView })));
const ModelsView = lazy(() => import("./views/ModelsView").then((m) => ({ default: m.ModelsView })));
const DevicesView = lazy(() => import("./views/DevicesView").then((m) => ({ default: m.DevicesView })));

export type ViewName = "train" | "datasets" | "generate" | "export" | "models" | "devices";

const NAV: { name: ViewName; label: string; icon: IconName }[] = [
  { name: "train", label: "Train", icon: "activity" },
  { name: "datasets", label: "Datasets", icon: "database" },
  { name: "generate", label: "Generate", icon: "sparkles" },
  { name: "export", label: "Export", icon: "upload" },
  { name: "models", label: "Models", icon: "box" },
  { name: "devices", label: "Devices", icon: "cpu" },
];

const VIEWS: Record<ViewName, ComponentType> = {
  train: TrainView,
  datasets: DatasetsView,
  generate: GenerateView,
  export: ExportView,
  models: ModelsView,
  devices: DevicesView,
};

function TokenGate() {
  const { submitToken } = useAppState();
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    const candidate = value.trim();
    if (!candidate || busy) return;
    setBusy(true);
    setError("");
    try {
      const ok = await submitToken(candidate);
      if (!ok) setError("Token rejected. Check it and try again.");
    } catch {
      setError("Could not reach the server.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen grid lg:grid-cols-[1.1fr_minmax(380px,0.9fr)]">
      <div className="relative hidden lg:flex flex-col justify-between p-12 xl:p-16 overflow-hidden border-r border-slate-800/60">
        <div className="absolute inset-0 opacity-40 [background-image:linear-gradient(rgba(241,232,216,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(241,232,216,0.04)_1px,transparent_1px)] [background-size:56px_56px]" />
        <div className="absolute -left-32 -top-32 h-[28rem] w-[28rem] rounded-full bg-amber-500/10 blur-3xl" />
        <div className="relative flex items-center gap-4">
          <CatLogo className="w-11 h-11" />
          <div>
            <p className="font-mono text-[9px] uppercase tracking-[0.34em] text-amber-300/70">
              Noctámbulo · MMXXVI
            </p>
            <p className="text-sm text-slate-400">Nocturnal training observatory</p>
          </div>
        </div>
        <div className="relative max-w-xl">
          <p className="mb-4 font-mono text-[10px] uppercase tracking-[0.3em] text-brass-light/70">
            Instrument access
          </p>
          <h1 className="font-display text-6xl xl:text-7xl leading-[0.95] text-ivory">
            Train while
            <br />
            the world sleeps.
          </h1>
          <p className="mt-6 max-w-md text-sm leading-relaxed text-slate-400">
            A precision console for language models: datasets, checkpoints, remote accelerators
            and loss curves on one dark instrument panel.
          </p>
        </div>
        <div className="relative flex items-center gap-6 font-mono text-[9px] uppercase tracking-[0.2em] text-slate-600">
          <span>WGPU</span>
          <span className="h-px w-6 bg-brass/50" />
          <span>Vulkan</span>
          <span className="h-px w-6 bg-brass/50" />
          <span>Remote</span>
        </div>
      </div>

      <div className="flex items-center justify-center p-6 sm:p-10">
        <div className="glass-card corner-tick w-full max-w-sm p-7 sm:p-8 rounded-xl animate-scale-in">
          <div className="mb-7 flex items-center gap-3 lg:hidden">
            <CatLogo className="w-10 h-10" />
            <div>
              <h1 className="text-lg font-semibold text-ivory tracking-tight">Noctámbulo</h1>
              <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-500">
                Training console
              </p>
            </div>
          </div>
          <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.26em] text-amber-300/70">
            Secure channel
          </p>
          <h2 className="font-display text-3xl text-ivory mb-2">Enter the observatory</h2>
          <p className="mb-6 text-xs leading-relaxed text-slate-500">
            Paste the API token printed by <span className="font-mono text-slate-400">serve</span>.
          </p>
          <label
            htmlFor="token-input"
            className="block font-mono text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-500 mb-2"
          >
            API Token
          </label>
          <input
            id="token-input"
            type="password"
            value={value}
            placeholder="••••••••••••"
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void submit();
            }}
            className="w-full px-3 py-2.5 rounded-md bg-slate-950/80 border border-slate-700/70 text-ivory text-sm placeholder-slate-600 focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 transition mb-4"
          />
          <button
            type="button"
            disabled={busy}
            onClick={() => void submit()}
            className="w-full py-2.5 rounded-md bg-gradient-to-b from-amber-300 to-amber-500 hover:from-amber-200 hover:to-amber-400 text-[#140d04] font-mono text-[11px] font-semibold uppercase tracking-[0.1em] shadow-[0_10px_30px_rgba(242,169,59,0.18)] transition active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
          >
            Connect
          </button>
          {error ? <p className="text-xs text-rose-400 mt-3">{error}</p> : null}
        </div>
      </div>
    </div>
  );
}

function ConnBadge() {
  const { conn } = useAppState();
  const styles =
    conn === "online"
      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
      : conn === "offline"
        ? "border-rose-500/30 bg-rose-500/10 text-rose-300"
        : "border-slate-700 bg-slate-900/80 text-slate-400";
  const dot =
    conn === "online"
      ? "bg-emerald-400 animate-pulse-dot"
      : conn === "offline"
        ? "bg-rose-400"
        : "bg-slate-500";
  const label = conn === "online" ? "online" : conn === "offline" ? "offline" : "connecting…";
  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.14em] px-2 py-1 rounded-[3px] border ${styles}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      <span>{label}</span>
    </span>
  );
}

const JOB_LABELS: Record<string, string> = {
  running: "training…",
  done: "completed",
  stopped: "stopped",
  error: "error",
};

function GlobalProgress() {
  const { job } = useAppState();
  const running = job.status === "running";
  const hasJob = running || job.status !== "idle";
  if (!hasJob) return null;
  const label = running
    ? job.paused
      ? "paused"
      : job.stopping
        ? "stopping…"
        : JOB_LABELS.running
    : (JOB_LABELS[job.status] ?? "—");
  const value = Number.isFinite(job.pct) ? job.pct : 0;
  const shimmer = running && !job.paused && !job.stopping;

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between gap-2 font-mono text-[9px] uppercase tracking-[0.12em] mb-1.5">
        <span className="text-slate-500 truncate">{label}</span>
        <span className="tnum text-amber-300/80">{value.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-solar-deep via-solar to-amber-300 transition-all duration-300 ${
            shimmer ? "progress-shimmer" : ""
          }`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function GlobalOps() {
  const { ops, cancelOp } = useAppState();
  const running = ops.filter((o) => o.status === "running").slice(0, 3);
  if (running.length === 0) return null;

  return (
    <div className="mt-3 space-y-2.5">
      {running.map((op) => {
        const pct = opPercent(op);
        return (
          <div key={op.id}>
            <div className="flex items-center justify-between gap-2 font-mono text-[9px] mb-1">
              <span className="text-slate-500 truncate">{op.label}</span>
              <button
                type="button"
                onClick={() => cancelOp(op.id)}
                className="text-rose-300/70 hover:text-rose-200 uppercase tracking-wider transition shrink-0"
              >
                cancel
              </button>
            </div>
            <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r from-solar-deep to-amber-300 ${
                  pct == null ? "progress-shimmer w-full" : ""
                }`}
                style={pct != null ? { width: `${pct}%` } : undefined}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StationBar({ view }: { view: ViewName }) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const index = NAV.findIndex((item) => item.name === view) + 1;
  const label = NAV.find((item) => item.name === view)?.label ?? view;
  const utc = now.toISOString().slice(11, 19);
  return (
    <div className="mb-5 hidden items-center justify-between border-b border-slate-800/50 pb-2.5 font-mono text-[8px] uppercase tracking-[0.22em] text-slate-600 lg:flex">
      <span>
        Station {String(index).padStart(2, "0")} / {label}
      </span>
      <span className="flex items-center gap-2">
        <span className="h-1 w-1 rounded-full bg-amber-400 animate-pulse-dot" />
        {utc} UTC · local console
      </span>
    </div>
  );
}

function Shell() {
  const { authState, logout } = useAppState();
  const [view, setView] = useState<ViewName>("train");
  const [drawerOpen, setDrawerOpen] = useState(false);

  if (authState === "gate") return <TokenGate />;
  if (authState === "checking") {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        <Icon name="loader" className="w-5 h-5 animate-spin" />
      </div>
    );
  }

  const ActiveView = VIEWS[view];

  const switchView = (name: ViewName) => {
    setDrawerOpen(false);
    setView(name);
  };

  return (
    <div className="min-h-screen flex">
      <button
        type="button"
        aria-label="Close menu"
        className={`fixed inset-0 z-30 bg-black/75 backdrop-blur-sm cursor-default lg:hidden ${
          drawerOpen ? "" : "hidden"
        }`}
        onClick={() => setDrawerOpen(false)}
      />

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 flex flex-col border-r border-slate-800/80 bg-slate-950/95 backdrop-blur-xl transition-transform duration-200 lg:translate-x-0 ${
          drawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-amber-500/[0.07] to-transparent" />
        <div className="relative flex items-center gap-3 px-5 pt-6 pb-5">
          <CatLogo className="w-10 h-10" />
          <div className="min-w-0">
            <h1 className="text-lg font-semibold text-ivory tracking-tight leading-tight">
              Noctámbulo
            </h1>
            <p className="font-mono text-[8px] uppercase tracking-[0.22em] text-brass-light/70">
              Training observatory
            </p>
          </div>
        </div>

        <div className="mx-5 mb-4 flex items-center gap-2">
          <span className="h-px flex-1 bg-slate-800" />
          <span className="font-mono text-[8px] uppercase tracking-[0.24em] text-slate-600">
            Instruments
          </span>
          <span className="h-px flex-1 bg-slate-800" />
        </div>

        <nav className="flex flex-col gap-0.5 px-3">
          {NAV.map(({ name, label, icon }, index) => {
            const active = name === view;
            return (
              <button
                key={name}
                type="button"
                data-index={String(index + 1).padStart(2, "0")}
                onClick={() => switchView(name)}
                className={`nav-index group relative flex w-full items-center gap-2 rounded-md px-3 py-2.5 font-mono text-[11px] uppercase tracking-[0.12em] transition ${
                  active
                    ? "nav-glow text-ivory"
                    : "text-slate-400 hover:text-ivory hover:bg-slate-900/70"
                }`}
              >
                <Icon name={icon} className="w-4 h-4 shrink-0 opacity-80" />
                <span>{label}</span>
                {active ? (
                  <span className="ml-auto h-1 w-1 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(242,169,59,0.8)]" />
                ) : null}
              </button>
            );
          })}
        </nav>

        <div className="flex-1" />

        <div className="relative m-3 rounded-lg border border-slate-800/80 bg-slate-950/70 p-3 corner-tick">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[8px] uppercase tracking-[0.22em] text-slate-600">
              Telemetry
            </span>
            <ConnBadge />
          </div>
          <GlobalProgress />
          <GlobalOps />
          <div className="mt-3 flex items-center justify-between border-t border-slate-800/70 pt-2.5">
            <span className="font-mono text-[8px] uppercase tracking-[0.18em] text-slate-600">
              v0.1 · local
            </span>
            <button
              type="button"
              onClick={logout}
              className="inline-flex items-center gap-1.5 font-mono text-[9px] uppercase tracking-[0.12em] text-slate-500 hover:text-amber-300 transition"
            >
              <Icon name="logout" className="w-3.5 h-3.5" />
              Log out
            </button>
          </div>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col lg:pl-64">
        <header className="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur sticky top-0 z-20">
          <button
            type="button"
            className="p-2 -m-2 text-slate-300 hover:text-amber-300 transition"
            aria-label="Open menu"
            onClick={() => setDrawerOpen(true)}
          >
            <Icon name="menu" className="w-5 h-5" />
          </button>
          <CatLogo className="w-7 h-7" />
          <h1 className="text-base font-semibold text-ivory tracking-tight">Noctámbulo</h1>
          <span className="ml-auto">
            <ConnBadge />
          </span>
        </header>

        <main className="flex-1 px-4 sm:px-6 lg:px-10 py-7 lg:py-9 max-w-7xl w-full mx-auto">
          <StationBar view={view} />
          <div key={view} className="animate-fade-slide">
            <Suspense
              fallback={
                <div className="flex justify-center py-16 text-slate-500">
                  <Icon name="loader" className="w-5 h-5 animate-spin" />
                </div>
              }
            >
              <ActiveView />
            </Suspense>
          </div>
        </main>

        <footer className="px-4 sm:px-6 lg:px-10 pb-5">
          <div className="mx-auto flex max-w-7xl items-center justify-between border-t border-slate-800/60 pt-3 font-mono text-[8px] uppercase tracking-[0.2em] text-slate-600">
            <span>Noctámbulo Engine · 2026</span>
            <span className="hidden sm:inline">local inference · remote accelerators</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <ModalProvider>
        <AppStateProvider>
          <Shell />
        </AppStateProvider>
      </ModalProvider>
    </ToastProvider>
  );
}
