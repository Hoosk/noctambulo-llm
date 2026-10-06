import { useEffect, useRef } from "react";
import { useAppState } from "../../state/appContext";

function lineClass(line: string): string {
  if (/error|panic|failed|fatal/i.test(line)) return "text-rose-400";
  if (/warn/i.test(line)) return "text-amber-300";
  if (/^== |checkpoint/i.test(line)) return "text-violet-300";
  return "text-slate-400";
}

export function LogPanel() {
  const { job } = useAppState();
  const nodeRef = useRef<HTMLPreElement | null>(null);

  // Lines arrive live over SSE (plus backfill on reconnect), so the panel only
  // needs to keep itself scrolled to the bottom.
  useEffect(() => {
    const node = nodeRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [job.logs]);

  return (
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
  );
}
