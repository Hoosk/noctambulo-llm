import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelInfo } from "../api/types";
import { Button } from "../components/Button";
import { ExportResult } from "../components/ExportResult";
import { Badge, ViewHeader } from "../components/ui";
import { fmtBytes } from "../lib/format";
import { useAppState } from "../state/appContext";
import { useToast } from "../state/toastContext";

function ModelRow({
  model,
  isActive,
  exporting,
  onUse,
  onExport,
}: {
  model: ModelInfo;
  isActive: boolean;
  exporting: boolean;
  onUse: () => void;
  onExport: () => void;
}) {
  return (
    <li
      className={`lift rounded-2xl border p-4 flex flex-wrap items-center gap-4 ${
        isActive ? "border-emerald-500/40 bg-emerald-500/5" : "border-slate-800/70 bg-slate-900/40"
      }`}
    >
      <div className="flex-1 min-w-52">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm text-white">{model.name}</span>
          {model.default ? <Badge tone="violet">default</Badge> : null}
          {isActive ? (
            <Badge tone="emerald" pulse>
              active
            </Badge>
          ) : null}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-500 tnum font-mono">
          <span>d {model.n_embd}</span>
          <span>heads {model.n_head}</span>
          <span>layers {model.n_layer}</span>
          <span>ctx {model.n_ctx}</span>
          <span>vocab {model.vocab}</span>
          <span>{model.params_m.toFixed(2)}M</span>
          <span>{fmtBytes(model.size_mb * 1024 * 1024)}</span>
        </div>
        <div className="mt-1 text-[10px] text-slate-600 font-mono truncate">{model.dir}</div>
      </div>
      <div className="flex gap-2 shrink-0">
        <Button variant="success" className="px-3 py-1.5 text-xs" disabled={isActive} onClick={onUse}>
          {isActive ? "Active ✓" : "Use"}
        </Button>
        <Button
          variant="primary"
          icon={exporting ? "loader" : "upload"}
          className={`px-3 py-1.5 text-xs ${exporting ? "[&_svg]:animate-spin" : ""}`}
          disabled={exporting}
          onClick={onExport}
        >
          {exporting ? "Exporting…" : "Export GGUF"}
        </Button>
      </div>
    </li>
  );
}

export function ModelsView() {
  const { activeModel, setActiveModel } = useAppState();
  const toast = useToast();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [exportingDir, setExportingDir] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getModels()
      .then((list) => {
        if (!cancelled) {
          setModels(list);
          setLoaded(true);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setLoaded(true);
          toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [toast]);

  const exportGguf = async (model: ModelInfo) => {
    if (exportingDir) return;
    setExportingDir(model.dir);
    try {
      const res = await api.exportModel(model.dir, model.name);
      toast.success(`GGUF generated: ${res.gguf}`);
      setResult(res.gguf);
    } catch (e) {
      toast.error(`Export failed: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setExportingDir(null);
    }
  };

  const active = models.find((m) => m.dir === activeModel);

  return (
    <>
      <ViewHeader
        title="Models"
        subtitle="Pick an active model for Generate/Export, or export to GGUF directly."
      />

      {active ? (
        <div className="mb-4 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-4 flex items-center gap-3 shadow-lg shadow-emerald-500/10">
          <span className="text-lg text-emerald-300">✓</span>
          <div className="flex-1 min-w-0">
            <div className="text-sm text-emerald-200 font-medium truncate">
              Active model: {active.name}
            </div>
            <div className="text-xs text-emerald-400/70 font-mono truncate">{active.dir}</div>
          </div>
          <Button onClick={() => setActiveModel("")}>Remove</Button>
        </div>
      ) : null}

      {loaded && models.length === 0 ? (
        <div className="p-10 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500">
          No models. Train one first.
        </div>
      ) : null}

      <ul className="mt-6 space-y-3">
        {models.map((m) => (
          <ModelRow
            key={m.dir}
            model={m}
            isActive={m.dir === activeModel}
            exporting={exportingDir === m.dir}
            onUse={() => setActiveModel(m.dir)}
            onExport={() => void exportGguf(m)}
          />
        ))}
      </ul>

      {result ? <ExportResult gguf={result} /> : null}
    </>
  );
}
