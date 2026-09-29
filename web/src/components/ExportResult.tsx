import { Icon } from "./Icon";

export function ExportResult({ gguf }: { gguf: string }) {
  return (
    <div className="mt-4 p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-emerald-500/60 text-xs flex flex-wrap items-center gap-3">
      <span className="text-emerald-300 font-medium flex-1 min-w-0 truncate font-mono">{gguf}</span>
      <a
        href={`/api/download/${encodeURIComponent(gguf)}`}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition active:scale-[0.98] shrink-0"
      >
        <Icon name="download" className="w-3.5 h-3.5" />
        Download GGUF
      </a>
    </div>
  );
}
