import { useEffect, useRef, useState } from "react";
import { Button } from "../../components/Button";
import { CatLogo } from "../../components/Icon";

export type StatusTone = "muted" | "running" | "ok" | "warn" | "error";

const STATUS_CLASSES: Record<StatusTone, string> = {
  muted: "text-slate-400",
  running: "text-amber-300",
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
  const [copiedText, setCopiedText] = useState("");

  useEffect(() => {
    const node = preRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [text]);

  const copy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedText(text);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="glass-card corner-tick rounded-xl p-5 min-w-0 lg:sticky lg:top-6 self-start">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          Output stream
        </h3>
        <Button
          variant="ghost"
          icon="copy"
          className="px-2.5 py-1 text-[10px]"
          disabled={!text}
          onClick={() => void copy()}
        >
          {copiedText === text ? "Copied" : "Copy"}
        </Button>
      </div>
      <div className="flex items-start gap-3">
        <div className="shrink-0 mt-1">
          <CatLogo className="w-8 h-8" />
        </div>
        <div className="flex-1 min-w-0 rounded-lg bg-slate-950/70 border border-slate-800/70 border-l-2 border-l-amber-500/60 p-4 min-h-28 max-h-[28rem] overflow-y-auto">
          {text ? (
            <pre
              ref={preRef}
              className="text-sm whitespace-pre-wrap text-slate-200 leading-relaxed font-sans"
            >
              {text}
            </pre>
          ) : (
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-600">
              Awaiting prompt…
            </p>
          )}
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 border-t border-slate-800/70 pt-2.5">
        <p className={`text-xs ${STATUS_CLASSES[status.tone]}`}>{status.text || "idle"}</p>
        <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-slate-600">
          autoregressive
        </span>
      </div>
    </div>
  );
}
