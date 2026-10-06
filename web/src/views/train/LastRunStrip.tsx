import { Badge } from "../../components/ui";
import { useAppState } from "../../state/appContext";
import { jobBadge } from "./jobBadge";
import { LogPanel } from "./LogPanel";

/** One-line summary of the last finished run, shown above the run history.
 * Keeps the idle view free of the full live-metrics card while still exposing
 * the final numbers and the worker logs. */
export function LastRunStrip() {
  const { job } = useAppState();
  if (job.status === "idle" || job.status === "running") return null;

  const badge = jobBadge(job);
  const lastLossPoint = job.lossPoints[job.lossPoints.length - 1];
  const lastValPoint = job.valPoints[job.valPoints.length - 1];
  const lastLoss = job.loss ?? lastLossPoint?.y ?? null;
  const lastVal = job.valLoss ?? lastValPoint?.y ?? null;
  const ppl = job.valPpl ?? (lastVal != null ? Math.exp(lastVal) : null);
  const steps = job.step || lastLossPoint?.x || 0;

  return (
    <div className="mb-4 rounded-xl border border-slate-800/70 bg-slate-950/40 px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Last run
        </span>
        <Badge tone={badge.tone}>{badge.label}</Badge>
        {steps > 0 ? <span className="tnum font-mono">steps {steps}</span> : null}
        {lastLoss != null ? <span className="tnum font-mono">loss {lastLoss.toFixed(4)}</span> : null}
        {ppl != null ? <span className="tnum font-mono">PPL {ppl.toFixed(1)}</span> : null}
      </div>
      <details className="mt-2">
        <summary className="text-[11px] text-slate-500 cursor-pointer select-none hover:text-slate-300 transition">
          Worker logs
        </summary>
        <LogPanel />
      </details>
    </div>
  );
}
