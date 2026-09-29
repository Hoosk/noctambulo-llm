import { useEffect, useRef } from "react";
import { CatLogo } from "../../components/Icon";

export type StatusTone = "muted" | "running" | "ok" | "warn" | "error";

const STATUS_CLASSES: Record<StatusTone, string> = {
  muted: "text-slate-400",
  running: "text-violet-300",
  ok: "text-emerald-300",
  warn: "text-amber-300",
  error: "text-rose-400",
};

export function GenerateOutput({
  text,
  status,
}: {
  text: string;
  status: { text: string; tone: StatusTone };
}) {
  const preRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    const node = preRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [text]);

  return (
    <div className="glass-card rounded-2xl p-5 min-w-0 lg:sticky lg:top-6 self-start">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">Output</h3>
      <div className="flex items-start gap-3">
        <div className="shrink-0 mt-1">
          <CatLogo className="w-8 h-8" />
        </div>
        <div className="flex-1 min-w-0 rounded-xl bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-violet-500/60 p-4 min-h-24 max-h-[28rem] overflow-y-auto">
          <pre
            ref={preRef}
            className="text-sm whitespace-pre-wrap text-slate-200 leading-relaxed font-sans"
          >
            {text || "—"}
          </pre>
        </div>
      </div>
      <p className={`text-xs mt-3 ${STATUS_CLASSES[status.tone]}`}>{status.text}</p>
    </div>
  );
}
