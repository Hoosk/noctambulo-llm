import { useEffect, useRef } from "react";
import type { JobInfo } from "../../state/jobModel";
import { LineChart, type ChartSeries } from "../../components/LineChart";
import { Badge, StatCard, type Tone } from "../../components/ui";
import type { IconName } from "../../components/Icon";
import { fmtDur } from "../../lib/format";
import { useAppState } from "../../state/appContext";

interface BadgeInfo {
  tone: Tone;
  label: string;
  pulse?: boolean;
}

interface StatItem {
  label: string;
  value: string;
  icon: IconName;
  tone: Tone;
}

function jobBadge(job: JobInfo): BadgeInfo {
  if (job.status === "running") {
    if (job.paused) return { tone: "amber", label: "paused" };
    if (job.stopping) return { tone: "amber", label: "stopping" };
    return { tone: "violet", label: "training", pulse: true };
  }
  if (job.status === "done") return { tone: "emerald", label: "completed" };
  if (job.status === "stopped") return { tone: "amber", label: "stopped" };
  if (job.status === "error") return { tone: "rose", label: "error" };
  return { tone: "slate", label: "idle" };
}

function jobStats(job: JobInfo): StatItem[] {
  const lossTone: Tone = job.loss == null ? "slate" : job.avgLoss != null && job.loss <= job.avgLoss ? "emerald" : "amber";
  return [
    { label: "Epoch", value: job.epochs ? `${job.epoch}/${job.epochs}` : "—", icon: "layers", tone: "violet" },
    { label: "Batch", value: job.numBatches ? `${job.batch}/${job.numBatches}` : "—", icon: "gauge", tone: "sky" },
    { label: "Context", value: job.nCtx ? `${job.nCtx} tok` : "—", icon: "cpu", tone: "sky" },
    { label: "Loss", value: job.loss != null ? job.loss.toFixed(4) : "—", icon: "activity", tone: lossTone },
    { label: "Avg Loss", value: job.avgLoss != null ? job.avgLoss.toFixed(4) : "—", icon: "gauge", tone: "slate" },
    { label: "Val PPL", value: job.valPpl != null ? job.valPpl.toFixed(2) : "—", icon: "zap", tone: "fuchsia" },
    { label: "Speed", value: job.batchesPerSec != null ? `${job.batchesPerSec.toFixed(1)} b/s` : "—", icon: "zap", tone: "sky" },
    { label: "ETA", value: fmtDur(job.etaSecs), icon: "clock", tone: "amber" },
    { label: "Elapsed", value: fmtDur(job.elapsedSecs), icon: "clock", tone: "slate" },
  ];
}

function lineClass(line: string): string {
  if (/error|panic|failed|fatal/i.test(line)) return "text-rose-400";
  if (/warn/i.test(line)) return "text-amber-300";
  if (/^== |checkpoint/i.test(line)) return "text-violet-300";
  return "text-slate-400";
}

function StatsGrid({ job }: { job: JobInfo }) {
  return (
    <div className="grid min-w-0 grid-cols-2 sm:grid-cols-3 gap-3 mb-5">
      {jobStats(job).map((s) => (
        <StatCard key={s.label} label={s.label} value={s.value} icon={s.icon} tone={s.tone} />
      ))}
    </div>
  );
}

function ProgressBar({ job }: { job: JobInfo }) {
  const running = job.status === "running";
  // Until the first Progress event the worker is still tokenizing the dataset;
  // there is no percentage yet, so show an indeterminate bar instead of 0%.
  const preparing = running && job.step === 0;
  const shimmer = running && !job.paused && !job.stopping;
  return (
    <div className="mt-5">
      <div className="flex items-center justify-between gap-3 mb-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          {preparing ? "Preparing dataset (tokenizing)…" : "Progress"}
        </h4>
        {preparing ? null : (
          <span className="text-[11px] text-slate-500 tnum font-mono">{job.pct.toFixed(1)}%</span>
        )}
      </div>
      <div className="h-2.5 rounded-full bg-slate-800/80 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300 ${
            shimmer ? "progress-shimmer" : ""
          } ${preparing ? "animate-pulse" : ""}`}
          style={{ width: preparing ? "100%" : `${job.pct || 0}%` }}
        />
      </div>
    </div>
  );
}

function PreviewPane({ text }: { text: string }) {
  return (
    <div className="mt-5">
      <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">Preview</h4>
      <pre className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-violet-500/60 text-xs whitespace-pre-wrap text-slate-300 min-h-16 leading-relaxed">
        {text || "—"}
      </pre>
    </div>
  );
}

function LogPanel() {
  const { job } = useAppState();
  const nodeRef = useRef<HTMLPreElement | null>(null);

  // Lines arrive live over SSE (plus backfill on reconnect), so the panel only
  // needs to keep itself scrolled to the bottom.
  useEffect(() => {
    const node = nodeRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [job.logs]);

  return (
    <details className="mt-5 group" open>
      <summary className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 cursor-pointer select-none hover:text-slate-300 transition">
        Worker logs
      </summary>
      <pre
        ref={nodeRef}
        className="mt-2 p-3 rounded-xl bg-black/40 border border-slate-800/70 text-[11px] font-mono leading-relaxed whitespace-pre-wrap h-44 overflow-y-auto"
      >
        {job.logs.map((entry) => (
          <span key={entry.id} className={`block ${lineClass(entry.text)}`}>
            {entry.text}
          </span>
        ))}
      </pre>
    </details>
  );
}

function lossSeries(job: JobInfo): ChartSeries[] {
  const series: ChartSeries[] = [
    { label: "train loss", points: job.lossPoints, color: "#8b5cf6" },
  ];
  if (job.valPoints.length > 0) {
    series.push({ label: "val loss", points: job.valPoints, color: "#f59e0b", dashed: true });
  }
  return series;
}

export function TrainMetrics() {
  const { job } = useAppState();
  const badge = jobBadge(job);

  return (
    <div className="glass-card rounded-2xl p-5 min-w-0">
      <div className="flex items-center justify-between gap-3 mb-5">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Live metrics</h3>
        <Badge tone={badge.tone} pulse={badge.pulse}>
          {badge.label}
        </Badge>
      </div>
      <StatsGrid job={job} />
      <LineChart series={lossSeries(job)} />
      <ProgressBar job={job} />
      <PreviewPane text={job.preview} />
      <LogPanel />
    </div>
  );
}
