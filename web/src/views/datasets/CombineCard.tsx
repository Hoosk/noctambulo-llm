import { useState } from "react";
import { api } from "../../api/client";
import { Button } from "../../components/Button";
import { OpProgress } from "../../components/OpProgress";
import { Field, Input } from "../../components/ui";
import { useAppState } from "../../state/appContext";
import { useToast } from "../../state/toastContext";
import { useOp, useOpOutcome } from "../../state/useOp";

type NoticeTone = "muted" | "ok" | "error";

const NOTICE_CLASSES: Record<NoticeTone, string> = {
  muted: "text-slate-400",
  ok: "text-emerald-300",
  error: "text-rose-400",
};

export function CombineCard({
  selectedNames,
  onDone,
}: {
  selectedNames: string[];
  onDone: () => void;
}) {
  const toast = useToast();
  const { cancelOp } = useAppState();
  const [weights, setWeights] = useState<Record<string, string>>({});
  const [output, setOutput] = useState("weighted_mix_web.txt");
  const [maxMb, setMaxMb] = useState("0");
  const [notice, setNotice] = useState<{ text: string; tone: NoticeTone }>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);
  const [opId, setOpId] = useState<string | null>(null);
  const op = useOp(opId);
  const running = op?.status === "running";

  useOpOutcome(opId, (o) => {
    if (o.status === "done") {
      const outputPath = typeof o.result?.output === "string" ? o.result.output : null;
      toast.success(o.message ?? "Mix created");
      setNotice({ text: outputPath ? `Mix created: ${outputPath}` : "Mix created", tone: "ok" });
      setWeights({});
      onDone();
    } else if (o.status === "cancelled") {
      setNotice({ text: "Mix cancelled", tone: "muted" });
    } else {
      setNotice({ text: `Error: ${o.error ?? "unknown"}`, tone: "error" });
    }
    setOpId(null);
  });

  const weightOf = (name: string) => weights[name] ?? "1";
  const parsedWeight = (name: string) => {
    const n = Number(weightOf(name));
    return Number.isFinite(n) && n > 0 ? n : 0;
  };
  const totalWeight = selectedNames.reduce((sum, name) => sum + parsedWeight(name), 0);
  const percentOf = (name: string) =>
    totalWeight > 0 ? `${((parsedWeight(name) / totalWeight) * 100).toFixed(0)}%` : "—";

  // Weighted mixing walks line by line; raw `.bin` token files cannot be mixed.
  const binaryNames = selectedNames.filter((name) => name.toLowerCase().endsWith(".bin"));
  const blockedReason =
    selectedNames.length < 2
      ? "Select at least 2 datasets in the table."
      : binaryNames.length > 0
        ? `Binary datasets cannot be mixed (${binaryNames.join(", ")}). Convert them to .txt first.`
        : "";

  const run = async () => {
    if (blockedReason || busy || running) return;
    const limitMb = Number(maxMb);
    setBusy(true);
    setNotice({ text: "Starting mix…", tone: "muted" });
    try {
      const res = await api.combineDatasets({
        sources: selectedNames.map((name) => ({ path: name, weight: parsedWeight(name) })),
        output: output.trim() || "weighted_mix_web.txt",
        maxChars: limitMb > 0 ? Math.round(limitMb * 1024 * 1024) : undefined,
      });
      setOpId(res.opId);
    } catch (e) {
      setNotice({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-5 mt-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Combine with weights</h3>
        <span className="text-xs text-slate-400 tnum">{selectedNames.length} selected</span>
      </div>
      <p className="text-xs text-slate-500 mb-2">
        Check datasets in the table above, adjust each weight and build a single mix. Weights are relative; the
        percentage shows each source&apos;s share of the result.
      </p>

      <div className="space-y-2 mt-3">
        {selectedNames.length < 2 ? (
          <p className="text-xs text-slate-500">Select at least 2 datasets from the table above.</p>
        ) : (
          <>
            <div className="grid grid-cols-[1fr_90px_70px] gap-3 text-[11px] font-semibold uppercase tracking-wide text-slate-600 px-1">
              <span>Dataset</span>
              <span>Weight</span>
              <span className="text-right">Share</span>
            </div>
            {selectedNames.map((name) => (
              <div key={name} className="grid grid-cols-[1fr_90px_70px] gap-3 items-center px-1 py-1">
                <span className="text-sm text-slate-200 font-mono truncate">{name}</span>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  value={weightOf(name)}
                  onChange={(e) => setWeights((prev) => ({ ...prev, [name]: e.target.value }))}
                  className="w-full px-2 py-1.5 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-xs tnum focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition"
                />
                <span className="text-xs text-slate-400 tnum font-mono text-right">{percentOf(name)}</span>
              </div>
            ))}
            <div className="flex justify-end px-1">
              <button
                type="button"
                onClick={() => setWeights({})}
                className="text-[11px] text-slate-400 hover:text-white transition underline decoration-dotted underline-offset-2"
              >
                Equalize weights
              </button>
            </div>
          </>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Mix name">
          <Input value={output} onChange={(e) => setOutput(e.target.value)} />
        </Field>
        <Field label="Max size (MB · 0 = all)">
          <Input type="number" min={0} step={64} value={maxMb} onChange={(e) => setMaxMb(e.target.value)} />
        </Field>
      </div>
      <div className="mt-4">
        <Button
          variant="success"
          icon="layers"
          disabled={busy || running || Boolean(blockedReason)}
          onClick={() => void run()}
        >
          {running ? "Combining…" : "Combine"}
        </Button>
      </div>
      {op && running ? (
        <OpProgress op={op} onCancel={() => cancelOp(op.id)} />
      ) : (
        <p className={`text-xs mt-3 ${blockedReason ? NOTICE_CLASSES.error : NOTICE_CLASSES[notice.tone]}`}>
          {blockedReason || notice.text}
        </p>
      )}
    </div>
  );
}
