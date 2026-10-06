import { useState } from "react";
import type { JobInfo } from "../../state/jobModel";
import { api } from "../../api/client";
import { Button } from "../../components/Button";
import { LineChart, type ChartSeries } from "../../components/LineChart";
import { Badge, StatCard, type Tone } from "../../components/ui";
import type { IconName } from "../../components/Icon";
import { fmtDur } from "../../lib/format";
import { useAppState } from "../../state/appContext";
import { useToast } from "../../state/toastContext";
import { jobBadge } from "./jobBadge";
import { LogPanel } from "./LogPanel";

interface StatItem {
  label: string;
  value: string;
  icon: IconName;
  tone: Tone;
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
  if (!running) return null;
  // Until the first Progress event the worker is still tokenizing the dataset;
  // there is no percentage yet, so show an indeterminate bar instead of 0%.
  const preparing = job.step === 0;
  const shimmer = !job.paused && !job.stopping;
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
  const toast = useToast();
  const [acting, setActing] = useState(false);
  const badge = jobBadge(job);
  const running = job.status === "running";

  const act = async (fn: () => Promise<unknown>, done: string) => {
    if (acting) return;
    setActing(true);
    try {
      await fn();
      toast.success(done);
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setActing(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-5 min-w-0">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div className="flex items-center gap-3">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Live metrics
          </h3>
          <Badge tone={badge.tone} pulse={badge.pulse}>
            {badge.label}
          </Badge>
        </div>
        {running ? (
          <div className="flex gap-2">
            <Button
              variant={job.paused ? "success" : "warn"}
              className="px-3 py-1.5 text-xs"
              disabled={job.stopping || acting}
              onClick={() =>
                void (job.paused
                  ? act(api.resumeTrain, "Resuming…")
                  : act(api.pauseTrain, "Pausing…"))
              }
            >
              {job.paused ? "Resume" : "Pause"}
            </Button>
            <Button
              variant="danger"
              className="px-3 py-1.5 text-xs"
              disabled={job.stopping || acting}
              onClick={() => void act(api.stopTrain, "Stopping…")}
            >
              {job.stopping ? "Stopping…" : "Stop"}
            </Button>
          </div>
        ) : null}
      </div>
      <StatsGrid job={job} />
      <LineChart series={lossSeries(job)} />
      <ProgressBar job={job} />
      <PreviewPane text={job.preview} />
      <details className="mt-5 group" open>
        <summary className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 cursor-pointer select-none hover:text-slate-300 transition">
          Worker logs
        </summary>
        <LogPanel />
      </details>
    </div>
  );
}
