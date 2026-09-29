import type { OpStatus } from "../api/types";
import { fmtBytes } from "../lib/format";
import { opPercent } from "../state/opsModel";

function detail(op: OpStatus): string {
  const unit = op.unit || "";
  if (unit === "bytes") return fmtBytes(op.done);
  const value = op.done.toLocaleString("en-US");
  return unit ? `${value} ${unit}` : value;
}

/** Progress bar for a running background operation, with optional cancel. */
export function OpProgress({ op, onCancel }: { op: OpStatus; onCancel?: () => void }) {
  const pct = opPercent(op);
  const label = op.label.length > 60 ? `${op.label.slice(0, 57)}…` : op.label;

  return (
    <div className="w-full mt-2">
      <div className="flex items-center justify-between gap-3 text-[11px] mb-1.5">
        <span className="text-slate-400 truncate">
          {label} — {op.phase}
        </span>
        <span className="tnum font-mono text-slate-500 shrink-0">
          {pct != null ? `${pct.toFixed(0)}%` : detail(op)}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300 ${
            pct == null ? "progress-shimmer w-full" : ""
          }`}
          style={pct != null ? { width: `${pct}%` } : undefined}
        />
      </div>
      {onCancel ? (
        <button
          type="button"
          onClick={onCancel}
          className="mt-1.5 text-[11px] text-rose-300/80 hover:text-rose-200 transition"
        >
          Cancel
        </button>
      ) : null}
    </div>
  );
}
