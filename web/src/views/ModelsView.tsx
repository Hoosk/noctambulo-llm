import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelFile, ModelInfo, OpStatus } from "../api/types";
import { Button } from "../components/Button";
import { Field, Input } from "../components/ui";
import { ExportResult } from "../components/ExportResult";
import { OpProgress } from "../components/OpProgress";
import { Badge, ViewHeader } from "../components/ui";
import { fmtBytes } from "../lib/format";
import { useAppState } from "../state/appContext";
import { useModal } from "../state/modalContext";
import { useToast } from "../state/toastContext";
import { useOp, useOpOutcome } from "../state/useOp";

function FilesPanel({ dir }: { dir: string }) {
  const [files, setFiles] = useState<ModelFile[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getModelFiles(dir)
      .then((res) => {
        if (!cancelled) setFiles(res.files);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "unknown");
      });
    return () => {
      cancelled = true;
    };
  }, [dir]);

  if (error) return <p className="w-full text-xs text-rose-400">Error: {error}</p>;
  if (!files) return <p className="w-full text-xs text-slate-500">Loading files…</p>;
  if (files.length === 0) return <p className="w-full text-xs text-slate-500">No files.</p>;

  return (
    <ul className="w-full divide-y divide-slate-800/60 rounded-xl border border-slate-800/60 bg-slate-950/40">
      {files.map((f) => (
        <li key={f.name} className="flex items-center gap-3 px-3 py-2 text-xs">
          <span className="font-mono text-slate-300 truncate">{f.name}</span>
          {f.shared ? <Badge tone="sky">shared</Badge> : null}
          <span className="ml-auto tnum text-slate-500">{fmtBytes(f.size_bytes)}</span>
          <a
            href={api.modelFileUrl(dir, f.name)}
            download
            className="text-violet-300 hover:text-violet-200 transition underline decoration-dotted underline-offset-2"
          >
            Download
          </a>
        </li>
      ))}
    </ul>
  );
}

function ModelRow({
  model,
  isActive,
  exporting,
  exportingOp,
  onUse,
  onExport,
  onCancelExport,
  onRename,
  onDelete,
}: {
  model: ModelInfo;
  isActive: boolean;
  exporting: boolean;
  exportingOp: OpStatus | null;
  onUse: () => void;
  onExport: () => void;
  onCancelExport: () => void;
  onRename: () => void;
  onDelete: () => void;
}) {
  const [showFiles, setShowFiles] = useState(false);

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
          {model.import_source ? <Badge tone="sky">{model.import_source}</Badge> : null}
          {model.compatible ? null : <Badge tone="rose">incompatible</Badge>}
          {model.steps != null && model.steps <= 2 ? <Badge tone="amber">demo</Badge> : null}
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
          <span>ffn {model.ffn_hidden}</span>
          <span>{model.params_m.toFixed(2)}M</span>
          <span>{fmtBytes(model.size_mb * 1024 * 1024)}</span>
        </div>
        <div className="mt-1 text-[10px] text-slate-600 font-mono truncate">{model.dir}</div>
        <div className="mt-0.5 text-[10px] text-slate-500 tnum">
          arch v{model.arch_version}
          {` · rope θ ${model.rope_base >= 1e5 ? model.rope_base.toExponential(0) : model.rope_base}`}
          {` · eps ${model.rms_norm_eps.toExponential(0)}`}
          {model.steps != null ? ` · ${model.steps} steps` : ""}
          {model.best_val_loss != null ? ` · best val ${model.best_val_loss.toFixed(3)}` : ""}
        </div>
      </div>
      <div className="flex gap-2 shrink-0 flex-wrap">
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
        <Button
          variant="ghost"
          icon="download"
          className="px-3 py-1.5 text-xs"
          aria-expanded={showFiles}
          onClick={() => setShowFiles((v) => !v)}
        >
          Files
        </Button>
        <Button variant="ghost" icon="pencil" className="px-3 py-1.5 text-xs" onClick={onRename}>
          Rename
        </Button>
        <Button
          variant="danger"
          icon="trash"
          className="px-3 py-1.5 text-xs"
          aria-label={`Delete ${model.name}`}
          onClick={onDelete}
        >
          Delete
        </Button>
      </div>
      {showFiles ? <FilesPanel dir={model.dir} /> : null}
      {exportingOp ? <OpProgress op={exportingOp} onCancel={onCancelExport} /> : null}
    </li>
  );
}

function ImportPanel({ onDone }: { onDone: () => void }) {
  const toast = useToast();
  const [ggufPath, setGgufPath] = useState("exported/");
  const [outputDir, setOutputDir] = useState("");
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!ggufPath.trim() || busy) return;
    setBusy(true);
    try {
      const res = await api.importModel({
        ggufPath: ggufPath.trim(),
        outputDir: outputDir.trim() || undefined,
        force,
      });
      toast.success(`Imported ${res.name}`);
      setOutputDir("");
      setForce(false);
      onDone();
    } catch (e) {
      toast.error(`Import failed: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mb-5 glass-card rounded-2xl p-4">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-3">
        Import GGUF
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="GGUF path (relative to the server)">
          <Input
            type="text"
            placeholder="exported/kuro-neko-v3.gguf"
            value={ggufPath}
            onChange={(e) => setGgufPath(e.target.value)}
          />
        </Field>
        <Field label="Output model name (optional)">
          <Input
            type="text"
            placeholder="derived from the file name"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
          />
        </Field>
      </div>
      <label className="mt-3 flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={force}
          onChange={(e) => setForce(e.target.checked)}
          className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
        />
        Overwrite if the model directory already exists
      </label>
      <div className="mt-3">
        <Button variant="primary" disabled={busy || !ggufPath.trim()} onClick={() => void run()}>
          {busy ? "Importing…" : "Import"}
        </Button>
      </div>
    </div>
  );
}

function ImportHfPanel({ onStarted }: { onStarted: (opId: string) => void }) {
  const toast = useToast();
  const [mode, setMode] = useState<"repo" | "dir">("repo");
  const [repo, setRepo] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [dir, setDir] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    const value = mode === "repo" ? repo.trim() : dir.trim();
    if (!value || busy) return;
    setBusy(true);
    try {
      const res = await api.importModelHf({
        repo: mode === "repo" ? value : undefined,
        dir: mode === "dir" ? value : undefined,
        outputDir: outputDir.trim() || undefined,
        force,
      });
      toast.info(`Converting ${res.name}…`);
      onStarted(res.opId);
    } catch (e) {
      toast.error(`Import failed: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mb-5 glass-card rounded-2xl p-4">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-3">
        Import from Hugging Face
      </h3>
      <div className="flex gap-2 mb-3">
        <Button
          variant={mode === "repo" ? "primary" : "ghost"}
          className="px-3 py-1.5 text-xs"
          onClick={() => setMode("repo")}
        >
          Hub repo
        </Button>
        <Button
          variant={mode === "dir" ? "primary" : "ghost"}
          className="px-3 py-1.5 text-xs"
          onClick={() => setMode("dir")}
        >
          Local directory
        </Button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {mode === "repo" ? (
          <Field label="Hugging Face repo id">
            <Input
              type="text"
              placeholder="Qwen/Qwen2.5-0.5B-Instruct"
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
            />
          </Field>
        ) : (
          <Field label="Directory with config.json + tokenizer.json + safetensors">
            <Input
              type="text"
              placeholder="downloads/my-model"
              value={dir}
              onChange={(e) => setDir(e.target.value)}
            />
          </Field>
        )}
        <Field label="Output model name (optional)">
          <Input
            type="text"
            placeholder="derived from the repo or directory"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
          />
        </Field>
      </div>
      <label className="mt-3 flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={force}
          onChange={(e) => setForce(e.target.checked)}
          className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
        />
        Overwrite if the model directory already exists
      </label>
      <p className="text-[11px] text-slate-500 mt-2">
        Supports Llama / Qwen2 / Mistral decoder models with SwiGLU, GQA, RMSNorm and a byte-level
        BPE tokenizer (tied embeddings, f32/f16/bf16 safetensors). Weights are converted to f32.
      </p>
      <div className="mt-3">
        <Button
          variant="primary"
          disabled={busy || (mode === "repo" ? !repo.trim() : !dir.trim())}
          onClick={() => void run()}
        >
          {busy ? "Starting…" : "Convert"}
        </Button>
      </div>
    </div>
  );
}

export function ModelsView() {
  const { activeModel, setActiveModel, cancelOp } = useAppState();
  const toast = useToast();
  const modal = useModal();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [exportState, setExportState] = useState<{ dir: string; opId: string } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [showHfImport, setShowHfImport] = useState(false);
  const [hfOpId, setHfOpId] = useState<string | null>(null);
  const exportOp = useOp(exportState?.opId ?? null);
  const hfOp = useOp(hfOpId);

  useOpOutcome(exportState?.opId ?? null, (op) => {
    if (op.status === "done") {
      const gguf = typeof op.result?.gguf === "string" ? op.result.gguf : null;
      toast.success(op.message ?? "GGUF exported");
      if (gguf) setResult(gguf);
    } else if (op.status === "cancelled") {
      toast.info("Export cancelled");
    } else {
      toast.error(`Export failed: ${op.error ?? "unknown"}`);
    }
    setExportState(null);
  });

  const load = useCallback(async () => {
    try {
      setModels(await api.getModels());
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setLoaded(true);
    }
  }, [toast]);

  useOpOutcome(hfOpId, (op) => {
    if (op.status === "done") {
      toast.success(op.message ?? "Model converted");
      void load();
    } else if (op.status === "cancelled") {
      toast.info("Conversion cancelled");
    } else {
      toast.error(`Conversion failed: ${op.error ?? "unknown"}`);
    }
    setHfOpId(null);
  });

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
    if (exportState) return;
    try {
      const res = await api.exportModel({ modelDir: model.dir, modelName: model.name });
      setExportState({ dir: model.dir, opId: res.opId });
    } catch (e) {
      toast.error(`Export failed: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const renameModel = async (model: ModelInfo) => {
    const newName = await modal.prompt({ title: "Rename model", label: "New name", value: model.name });
    if (!newName || newName.trim() === model.name) return;
    try {
      const res = await api.renameModel(model.dir, newName.trim());
      if (activeModel === model.dir) setActiveModel(res.dir);
      toast.success(`Renamed to ${res.name ?? newName.trim()}`);
      await load();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const deleteModel = async (model: ModelInfo) => {
    const ok = await modal.confirm({
      title: "Delete model?",
      message: `${model.name} — weights, config and optimizer state will be removed. Exported GGUF files are kept.`,
      confirmText: "Delete",
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteModel(model.name);
      if (activeModel === model.dir) setActiveModel("");
      toast.success(`${model.name} deleted`);
      await load();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const active = models.find((m) => m.dir === activeModel);

  return (
    <>
      <ViewHeader
        title="Models"
        subtitle="Pick an active model for Generate/Export, download its files, rename or delete it."
      />

      <div className="mb-4 flex flex-wrap gap-2">
        <Button variant="ghost" onClick={() => setShowImport((v) => !v)}>
          {showImport ? "Hide import" : "Import GGUF"}
        </Button>
        <Button variant="ghost" onClick={() => setShowHfImport((v) => !v)}>
          {showHfImport ? "Hide HF import" : "Import from Hugging Face"}
        </Button>
      </div>
      {showImport ? <ImportPanel onDone={() => void load()} /> : null}
      {showHfImport ? <ImportHfPanel onStarted={setHfOpId} /> : null}
      {hfOp ? (
        <div className="mb-4">
          <OpProgress op={hfOp} onCancel={() => cancelOp(hfOp.id)} />
        </div>
      ) : null}

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
            exporting={exportState?.dir === m.dir}
            exportingOp={exportState?.dir === m.dir ? exportOp : null}
            onUse={() => setActiveModel(m.dir)}
            onExport={() => void exportGguf(m)}
            onCancelExport={() => {
              if (exportOp) cancelOp(exportOp.id);
            }}
            onRename={() => void renameModel(m)}
            onDelete={() => void deleteModel(m)}
          />
        ))}
      </ul>

      {result ? <ExportResult gguf={result} /> : null}
    </>
  );
}
