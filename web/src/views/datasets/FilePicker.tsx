import { useMemo, useState } from "react";
import { Input } from "../../components/ui";

interface FilePickerProps {
  files: string[];
  /** Selected file; "" means the auto-picked best training file. */
  value: string;
  onChange: (file: string) => void;
  /** When set, an "Auto" entry is shown with this label. */
  autoLabel?: string;
  /** Max rows rendered; the rest need a more specific filter. */
  limit?: number;
}

// Data files first: repos with thousands of shards stay usable.
const EXT_ORDER = [".parquet", ".txt", ".jsonl", ".json", ".csv", ".arrow", ".bin"];

function rank(file: string): number {
  const lower = file.toLowerCase();
  const index = EXT_ORDER.findIndex((ext) => lower.endsWith(ext));
  return index === -1 ? EXT_ORDER.length : index;
}

export function FilePicker({ files, value, onChange, autoLabel, limit = 300 }: FilePickerProps) {
  const [filter, setFilter] = useState("");
  const sorted = useMemo(
    () => [...files].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)),
    [files],
  );
  const needle = filter.trim().toLowerCase();
  const filtered = needle ? sorted.filter((f) => f.toLowerCase().includes(needle)) : sorted;
  const shown = filtered.slice(0, limit);

  const itemClass = (selected: boolean) =>
    `w-full text-left px-2.5 py-1.5 rounded-lg transition truncate ${
      selected
        ? "bg-violet-500/20 text-violet-200"
        : "text-slate-300 hover:bg-slate-800/60"
    }`;

  return (
    <div className="rounded-xl border border-slate-800/70 bg-slate-950/40">
      <div className="p-2 border-b border-slate-800/60">
        <Input
          placeholder={`Filter ${files.length} files…`}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <ul className="max-h-52 overflow-y-auto p-1 space-y-0.5">
        {autoLabel !== undefined ? (
          <li>
            <button type="button" className={itemClass(value === "")} onClick={() => onChange("")}>
              <span className="font-mono text-xs">{autoLabel}</span>
            </button>
          </li>
        ) : null}
        {shown.map((file) => (
          <li key={file}>
            <button
              type="button"
              title={file}
              className={itemClass(value === file)}
              onClick={() => onChange(file)}
            >
              <span className="font-mono text-xs">{file}</span>
            </button>
          </li>
        ))}
        {filtered.length === 0 ? (
          <li className="px-2.5 py-2 text-xs text-slate-500">No files match.</li>
        ) : null}
        {filtered.length > shown.length ? (
          <li className="px-2.5 py-1.5 text-[10px] text-slate-500">
            … {filtered.length - shown.length} more; type to narrow down
          </li>
        ) : null}
      </ul>
    </div>
  );
}
