import { lazy, Suspense, useState } from "react";
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
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="glass-card max-w-sm w-full p-8 rounded-2xl animate-scale-in">
        <div className="flex items-center gap-4 mb-8">
          <CatLogo className="w-12 h-12" />
          <div className="min-w-0">
            <h1 className="text-lg font-semibold text-white tracking-tight">Noctámbulo</h1>
            <p className="text-xs text-slate-500">Training console</p>
          </div>
        </div>
        <label
          htmlFor="token-input"
          className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2"
        >
          API Token
        </label>
        <input
          id="token-input"
          type="password"
          value={value}
          placeholder="Paste your token here"
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
          className="w-full px-3 py-2.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition mb-4"
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => void submit()}
          className="w-full py-2.5 rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium shadow-lg shadow-violet-600/25 transition active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
        >
          Connect
        </button>
        {error ? <p className="text-xs text-rose-400 mt-3">{error}</p> : null}
      </div>
    </div>
  );
}

function ConnBadge() {
  const { conn } = useAppState();
  const styles =
    conn === "online"
      ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-300"
      : conn === "offline"
        ? "border-rose-500/30 bg-rose-500/15 text-rose-300"
        : "border-slate-700 bg-slate-800/80 text-slate-400";
  const dot =
    conn === "online"
      ? "bg-emerald-400 animate-pulse-dot"
      : conn === "offline"
        ? "bg-rose-400"
        : "bg-slate-500";
  const label = conn === "online" ? "online" : conn === "offline" ? "offline" : "connecting…";
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-1 rounded-full border ${styles}`}
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
      <div className="flex items-center justify-between gap-2 text-[11px] mb-1.5">
        <span className="text-slate-500 truncate">{label}</span>
        <span className="tnum font-mono text-slate-400">{value.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300 ${
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
            <div className="flex items-center justify-between gap-2 text-[10px] mb-1">
              <span className="text-slate-500 truncate">{op.label}</span>
              <button
                type="button"
                onClick={() => cancelOp(op.id)}
                className="text-rose-300/70 hover:text-rose-200 transition shrink-0"
              >
                cancel
              </button>
            </div>
            <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
              <div
                className={`h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 ${
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
        className={`fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-sm cursor-default lg:hidden ${
          drawerOpen ? "" : "hidden"
        }`}
        onClick={() => setDrawerOpen(false)}
      />

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 flex flex-col p-4 glass-card rounded-none border-y-0 border-l-0 transition-transform duration-200 lg:translate-x-0 ${
          drawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 px-1 mb-6">
          <CatLogo className="w-10 h-10" />
          <div className="min-w-0">
            <h1 className="text-base font-semibold text-white tracking-tight leading-tight">
              Noctámbulo
            </h1>
            <p className="text-[11px] text-slate-500">Training console</p>
          </div>
        </div>

        <nav className="flex flex-col gap-1">
          {NAV.map(({ name, label, icon }) => {
            const active = name === view;
            return (
              <button
                key={name}
                type="button"
                onClick={() => switchView(name)}
                className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  active
                    ? "nav-glow text-white font-medium"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                }`}
              >
                <Icon name={icon} className="w-4 h-4 shrink-0" />
                <span>{label}</span>
              </button>
            );
          })}
        </nav>

        <div className="flex-1" />

        <div className="rounded-xl border border-slate-800/70 bg-slate-950/60 p-3">
          <GlobalProgress />
          <GlobalOps />
          <div className="flex items-center justify-center">
            <ConnBadge />
          </div>
          <div className="mt-2 flex justify-center">
            <button
              type="button"
              onClick={logout}
              className="inline-flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-white transition"
            >
              <Icon name="logout" className="w-3.5 h-3.5" />
              Log out
            </button>
          </div>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col lg:pl-64">
        <header className="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-slate-800/70 bg-slate-950/60 backdrop-blur sticky top-0 z-20">
          <button
            type="button"
            className="p-2 -m-2 text-slate-300 hover:text-white transition"
            aria-label="Open menu"
            onClick={() => setDrawerOpen(true)}
          >
            <Icon name="menu" className="w-5 h-5" />
          </button>
          <CatLogo className="w-7 h-7" />
          <h1 className="text-base font-semibold text-white tracking-tight">Noctámbulo</h1>
          <span className="ml-auto">
            <ConnBadge />
          </span>
        </header>

        <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 max-w-6xl w-full mx-auto">
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
