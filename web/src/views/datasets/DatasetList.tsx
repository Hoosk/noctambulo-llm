import { useEffect, useRef, useState } from "react";
import type { DatasetInfo } from "../../api/types";
import { Badge } from "../../components/ui";
import { fmtBytes } from "../../lib/format";

const ROW_BTN =
  "px-2 py-1 rounded-md text-[11px] font-medium bg-slate-800/70 hover:bg-slate-700/70 text-slate-300 border border-slate-700/50 transition active:scale-[0.98]";

interface Props {
  datasets: DatasetInfo[];
  selected: Set<string>;
  onToggle: (name: string, checked: boolean) => void;
  onToggleAll: (checked: boolean) => void;
  onInfo: (dataset: DatasetInfo) => void;
  onRename: (dataset: DatasetInfo) => void;
  onDelete: (dataset: DatasetInfo) => void;
}

function DatasetRow({
  dataset,
  checked,
  onToggle,
  onInfo,
  onRename,
  onDelete,
}: {
  dataset: DatasetInfo;
  checked: boolean;
  onToggle: (checked: boolean) => void;
  onInfo: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="lift grid grid-cols-[auto_1fr_auto_auto] gap-3 items-center px-4 py-2.5 rounded-xl bg-slate-950/70 border border-slate-800/70">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onToggle(e.target.checked)}
        className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
      />
      <span className="flex items-center gap-2 text-sm text-slate-200 font-mono truncate">
        <Badge tone={dataset.ext === "bin" ? "violet" : "sky"}>{dataset.ext}</Badge>
        {dataset.name}
      </span>
      <span className="text-xs text-slate-500 tnum font-mono">{fmtBytes(dataset.size_bytes)}</span>
      <div className="flex gap-1.5">
        <button type="button" className={ROW_BTN} onClick={onInfo}>
          info
        </button>
        <button type="button" className={ROW_BTN} onClick={onRename}>
          rename
        </button>
        <button
          type="button"
          className={`${ROW_BTN} bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border-rose-500/20`}
          onClick={onDelete}
        >
          delete
        </button>
      </div>
    </div>
  );
}

function SelectAllCheckbox({
  checked,
  indeterminate,
  onChange,
}: {
  checked: boolean;
  indeterminate: boolean;
  onChange: (checked: boolean) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = indeterminate;
  }, [indeterminate]);
  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
    />
  );
}

export function DatasetList({ datasets, selected, onToggle, onToggleAll, onInfo, onRename, onDelete }: Props) {
  const [query, setQuery] = useState("");

  const totalBytes = datasets.reduce((acc, d) => acc + (d.size_bytes || 0), 0);
  const allSelected = datasets.length > 0 && datasets.every((d) => selected.has(d.name));
  const visible = query
    ? datasets.filter((d) => d.name.toLowerCase().includes(query.toLowerCase()))
    : datasets;

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Local datasets</h3>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 tnum">
            {datasets.length} datasets · {fmtBytes(totalBytes)}
          </span>
          <input
            type="text"
            placeholder="Filter by name…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-56 max-w-full px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-xs placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition"
          />
        </div>
      </div>
      <div className="grid grid-cols-[auto_1fr_auto_auto] gap-3 px-4 py-2 text-[11px] font-semibold uppercase tracking-wide text-slate-600 border-b border-slate-800/70">
        <SelectAllCheckbox checked={allSelected} indeterminate={!allSelected && selected.size > 0} onChange={onToggleAll} />
        <span>Name</span>
        <span>Size</span>
        <span>Actions</span>
      </div>
      <div className="space-y-2 mt-3">
        {visible.length === 0 ? (
          <p className="text-xs text-slate-500 px-4 py-6 text-center">No matches.</p>
        ) : (
          visible.map((d) => (
            <DatasetRow
              key={d.name}
              dataset={d}
              checked={selected.has(d.name)}
              onToggle={(checked) => onToggle(d.name, checked)}
              onInfo={() => onInfo(d)}
              onRename={() => onRename(d)}
              onDelete={() => onDelete(d)}
            />
          ))
        )}
      </div>
    </div>
  );
}
