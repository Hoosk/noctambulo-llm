import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { RunRecord, RunSummary } from "../../api/types";
import { Button } from "../../components/Button";
import { LineChart, type ChartSeries } from "../../components/LineChart";
import { Badge, type Tone } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useAppState } from "../../state/appContext";
import { LastRunStrip } from "./LastRunStrip";

const COLORS = ["#8b5cf6", "#38bdf8", "#f59e0b", "#34d399", "#fb7185"];
const MAX_SELECTED = 3;

function statusTone(status: string): Tone {
  if (status === "done") return "emerald";
  if (status === "running") return "violet";
  if (status === "failed") return "rose";
  return "amber";
}

function fmtDate(unix: number): string {
  return new Date(unix * 1000).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function RunHistoryCard() {
  const { job } = useAppState();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [records, setRecords] = useState<Record<string, RunRecord>>({});
  const [loaded, setLoaded] = useState(false);
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  // Initial history load.
  useEffect(() => {
    let cancelled = false;
    api
      .getRuns()
      .then((list) => {
        if (!cancelled) {
          setRuns(list);
          setLoaded(true);
        }
      })
      .catch(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A finished job finalizes its run server-side; refresh once it is terminal.
  useEffect(() => {
    if (job.status !== "done" && job.status !== "stopped" && job.status !== "error") return;
    let cancelled = false;
    api
      .getRuns()
      .then((list) => {
        if (!cancelled) setRuns(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [job.status]);

  // Fetch curves for newly selected runs (cache keeps re-renders cheap).
  useEffect(() => {
    const missing = selected.filter((id) => !records[id]);
    if (missing.length === 0) return;
    let cancelled = false;
    Promise.all(missing.map((id) => api.getRun(id)))
      .then((list) => {
        if (cancelled) return;
        setRecords((prev) => {
          const next = { ...prev };
          for (const record of list) next[record.id] = record;
          return next;
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [selected, records]);

  const toggleRun = (id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= MAX_SELECTED) return prev;
      return [...prev, id];
    });
  };

  const series: ChartSeries[] = selected.flatMap((id, i) => {
    const record = records[id];
    if (!record) return [];
    return [
      {
        label: `${record.dataset} · ${fmtDate(record.started_unix)}`,
        points: record.loss_points,
        color: COLORS[i % COLORS.length],
      },
    ];
  });

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Run history</h3>
        <Button
          icon="refresh"
          className="px-3 py-1.5 text-xs"
          onClick={() => {
            api
              .getRuns()
              .then(setRuns)
              .catch(() => {});
          }}
        >
          Refresh
        </Button>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Select up to {MAX_SELECTED} runs to overlay their training loss curves.
      </p>

      <LastRunStrip />

      {loaded && runs.length === 0 ? (
        <div className="p-8 rounded-xl border border-dashed border-slate-700/70 text-center text-slate-500 text-sm">
          No runs yet. Start a training job and it will show up here.
        </div>
      ) : null}

      {runs.length > 0 ? (
        <ul className="max-h-72 overflow-y-auto divide-y divide-slate-800/60 rounded-xl border border-slate-800/60 bg-slate-950/40">
          {runs.map((run) => {
            const checked = selectedSet.has(run.id);
            return (
              <li key={run.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2.5">
                <label className="flex items-center gap-2 cursor-pointer min-w-52">
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!checked && selected.length >= MAX_SELECTED}
                    onChange={() => toggleRun(run.id)}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-200 font-mono truncate">{run.dataset}</span>
                </label>
                <Badge tone={statusTone(run.status)}>{run.status}</Badge>
                <span className="text-xs text-slate-500">{fmtDate(run.started_unix)}</span>
                <span className="text-xs text-slate-500 tnum font-mono">
                  d {run.n_embd} · L {run.n_layer} · ctx {run.n_ctx} · {run.dtype} · ×{run.grad_accum}
                </span>
                <span className="ml-auto text-xs text-slate-400 tnum font-mono">
                  loss {fmtNum(run.final_loss, 3)} · PPL {fmtNum(run.final_ppl, 1)}
                </span>
              </li>
            );
          })}
        </ul>
      ) : null}

      {series.length > 0 ? (
        <div className="mt-4">
          <LineChart series={series} height={280} />
        </div>
      ) : null}
    </div>
  );
}
