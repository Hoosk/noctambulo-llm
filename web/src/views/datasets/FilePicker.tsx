import { useMemo, useState } from "react";
import { Button } from "../../components/Button";
import { Input } from "../../components/ui";

interface FilePickerProps {
  files: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
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

export function FilePicker({ files, selected, onChange, limit = 300 }: FilePickerProps) {
  const [filter, setFilter] = useState("");
  const sorted = useMemo(
    () => [...files].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)),
    [files],
  );
  const needle = filter.trim().toLowerCase();
  const filtered = needle ? sorted.filter((f) => f.toLowerCase().includes(needle)) : sorted;
  const shown = filtered.slice(0, limit);
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const toggle = (file: string) => {
    if (selectedSet.has(file)) onChange(selected.filter((f) => f !== file));
    else onChange([...selected, file]);
  };
  const selectAllFiltered = () => onChange([...new Set([...selected, ...filtered])]);

  return (
    <div className="rounded-xl border border-slate-800/70 bg-slate-950/40">
      <div className="p-2 border-b border-slate-800/60 flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <Input
            placeholder={`Filter ${files.length} files…`}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
        <Button
          variant="ghost"
          className="px-2.5 py-1.5 text-xs shrink-0"
          onClick={selectAllFiltered}
          disabled={filtered.length === 0}
        >
          Select all{needle ? " filtered" : ""}
        </Button>
        {selected.length > 0 ? (
          <Button variant="ghost" className="px-2.5 py-1.5 text-xs shrink-0" onClick={() => onChange([])}>
            Clear
          </Button>
        ) : null}
      </div>
      <ul className="max-h-52 overflow-y-auto p-1 space-y-0.5">
        {shown.map((file) => (
          <li key={file}>
            <label className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg hover:bg-slate-800/60 cursor-pointer">
              <input
                type="checkbox"
                className="accent-violet-500 cursor-pointer shrink-0"
                checked={selectedSet.has(file)}
                onChange={() => toggle(file)}
              />
              <span className="font-mono text-xs truncate text-slate-300" title={file}>
                {file}
              </span>
            </label>
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
      <div className="px-2.5 py-1.5 border-t border-slate-800/60 text-[10px] text-slate-500 tnum">
        {selected.length} file{selected.length === 1 ? "" : "s"} selected
      </div>
    </div>
  );
}
