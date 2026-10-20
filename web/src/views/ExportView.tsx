import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelInfo } from "../api/types";
import { Button } from "../components/Button";
import { ExportResult } from "../components/ExportResult";
import { OpProgress } from "../components/OpProgress";
import { Field, Input, Select, ViewHeader } from "../components/ui";
import { useAppState } from "../state/appContext";
import { useToast } from "../state/toastContext";
import { useOp, useOpOutcome } from "../state/useOp";

export function ExportView() {
  const { activeModel, cancelOp } = useAppState();
  const toast = useToast();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelDir, setModelDir] = useState(activeModel);
  const [name, setName] = useState("noctambulo");
  const [approx, setApprox] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [opId, setOpId] = useState<string | null>(null);
  const op = useOp(opId);
  const running = op?.status === "running";

  useOpOutcome(opId, (outcome) => {
    if (outcome.status === "done") {
      const gguf = typeof outcome.result?.gguf === "string" ? outcome.result.gguf : null;
      toast.success(outcome.message ?? "GGUF exported");
      if (gguf) setResult(gguf);
    } else if (outcome.status === "cancelled") {
      toast.info("Export cancelled");
    } else {
      setError(outcome.error ?? "unknown");
    }
    setOpId(null);
  });

  useEffect(() => {
    let cancelled = false;
    api
      .getModels()
      .then((list) => {
        if (cancelled) return;
        setModels(list);
        // Self-heal a stale persisted selection (model deleted/renamed outside
        // the UI) so Export never posts a non-existent model dir.
        setModelDir((current) =>
          current && list.some((m) => m.dir === current)
            ? current
            : (list.find((m) => m.default)?.dir ?? list[0]?.dir ?? ""),
        );
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "unknown");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedDir = models.some((m) => m.dir === modelDir)
    ? modelDir
    : (models.find((m) => m.default)?.dir ?? models[0]?.dir ?? "");

  const run = async () => {
    if (running || !selectedDir) return;
    setError("");
    setResult(null);
    try {
      const res = await api.exportModel({
        modelDir: selectedDir,
        modelName: name || "noctambulo",
        approx,
      });
      setOpId(res.opId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "unknown");
    }
  };

  return (
    <>
      <ViewHeader
        title="Export to GGUF"
        subtitle="Convert a trained model to GGUF for llama.cpp and other runtimes."
      />
      <div className="max-w-md glass-card rounded-2xl p-5">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">
          Configuration
        </h3>
        <div className="mb-3">
          <Field label="Model">
            <Select value={selectedDir} onChange={(e) => setModelDir(e.target.value)}>
              {models.length === 0 ? (
                <option value="">no models</option>
              ) : (
                models.map((m) => (
                  <option key={m.dir} value={m.dir}>
                    {m.name} · {m.params_m.toFixed(1)}M
                  </option>
                ))
              )}
            </Select>
          </Field>
        </div>
        <div className="mb-4">
          <Field label="GGUF name">
            <Input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
        </div>
        <label className="flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none mb-4">
          <input
            type="checkbox"
            checked={approx}
            onChange={(e) => setApprox(e.target.checked)}
            className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
          />
          Approx export (standard-attention approximation for grouped KV models)
        </label>
        <Button variant="primary" disabled={running || !selectedDir} onClick={() => void run()}>
          {running ? "Exporting…" : "Export"}
        </Button>
        {op && running ? <OpProgress op={op} onCancel={() => cancelOp(op.id)} /> : null}
        <p className="text-xs text-rose-400 mt-3">{error}</p>
        {result ? <ExportResult gguf={result} /> : null}
      </div>
    </>
  );
}
