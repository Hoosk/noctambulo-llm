import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { CheckpointInfo, DatasetInfo, DeviceInfo, TrainRequestBody } from "../../api/types";
import { Button } from "../../components/Button";
import { Field, Input, Select } from "../../components/ui";
import { fmtBytes } from "../../lib/format";
import { useAppState } from "../../state/appContext";

interface FormState {
  file: string;
  device: string;
  dtype: string;
  earlyStop: string;
  epochs: string;
  batch: string;
  lr: string;
  maxTokens: string;
  preset: string;
  nCtx: string;
  nEmbd: string;
  nHead: string;
  nLayer: string;
  valSplit: string;
  gradAccum: string;
  nKvHead: string;
  dropout: string;
  attnSoftcap: string;
  slidingWindow: string;
  sinkTokens: string;
  kvShareGroup: string;
  mlaLatentDim: string;
  kvEviction: string;
  kvBudget: string;
  kvRecent: string;
  seqWarmup: string;
  weightDecay: string;
  labelSmoothing: string;
  zLoss: string;
  saveDir: string;
  outputDir: string;
}

const PRESETS: Record<string, { nEmbd: string; nHead: string; nKvHead: string; nLayer: string; nCtx: string }> = {
  nano: { nEmbd: "128", nHead: "4", nKvHead: "2", nLayer: "4", nCtx: "256" },
  tiny: { nEmbd: "256", nHead: "4", nKvHead: "2", nLayer: "4", nCtx: "256" },
  small: { nEmbd: "512", nHead: "8", nKvHead: "2", nLayer: "8", nCtx: "512" },
  base: { nEmbd: "768", nHead: "12", nKvHead: "4", nLayer: "12", nCtx: "1024" },
};

const DEFAULT_FORM: FormState = {
  file: "",
  device: "DefaultDevice",
  dtype: "f32",
  earlyStop: "0",
  epochs: "1",
  batch: "32",
  lr: "0.00005",
  maxTokens: "10000000",
  preset: "tiny",
  nCtx: "256",
  nEmbd: "256",
  nHead: "4",
  nLayer: "4",
  valSplit: "0.02",
  gradAccum: "1",
  nKvHead: "2",
  dropout: "0.1",
  attnSoftcap: "0",
  slidingWindow: "0",
  sinkTokens: "0",
  kvShareGroup: "1",
  mlaLatentDim: "0",
  kvEviction: "none",
  kvBudget: "0",
  kvRecent: "64",
  seqWarmup: "0",
  weightDecay: "0.1",
  labelSmoothing: "0",
  zLoss: "0",
  saveDir: "checkpoints_web",
  outputDir: "model_output",
};

type NoticeTone = "muted" | "ok" | "warn" | "error";

const NOTICE_CLASSES: Record<NoticeTone, string> = {
  muted: "text-slate-400",
  ok: "text-emerald-300",
  warn: "text-amber-300",
  error: "text-rose-400",
};

const STEPS = ["Data", "Model", "Train", "Review"] as const;

function hiddenDim(nEmbd: number): number {
  return Math.ceil(Math.ceil((8 * nEmbd) / 3) / 256) * 256;
}

/** Mirrors ModelConfig::param_count so the wizard shows the real size. */
function paramCount(f: FormState): number {
  const nEmbd = Number(f.nEmbd) || 0;
  const nHead = Number(f.nHead) || 1;
  const nKvHead = Number(f.nKvHead) || nHead;
  const nLayer = Number(f.nLayer) || 0;
  const vocab = 8000;
  const kvDim = Math.max(1, Math.floor(nKvHead * (nEmbd / Math.max(1, nHead))));
  const hidden = hiddenDim(nEmbd);
  const perLayer = nEmbd * nEmbd + nEmbd + nEmbd * nEmbd + 2 * nEmbd + 3 * nEmbd * hidden;
  const mla = Number(f.mlaLatentDim) || 0;
  const group = Math.max(1, Number(f.kvShareGroup) || 1);
  const groups = Math.ceil(nLayer / group);
  const perGroup = mla > 0 ? nEmbd * mla + 2 * kvDim * mla : 2 * (nEmbd * kvDim + kvDim);
  return vocab * nEmbd + nLayer * perLayer + groups * perGroup + nEmbd;
}

function validateStep(step: number, f: FormState, datasets: DatasetInfo[]): string | null {
  if (step === 0) {
    const file = f.file || datasets[0]?.name || "";
    if (!file) return "Select a dataset to train on.";
    if (!f.outputDir.trim()) return "Set an output directory for the finished model.";
    if (!f.saveDir.trim()) return "Set a checkpoint directory.";
  }
  if (step === 1) {
    const nEmbd = Number(f.nEmbd);
    const nHead = Number(f.nHead);
    const nKvHead = Number(f.nKvHead);
    const nLayer = Number(f.nLayer);
    const nCtx = Number(f.nCtx);
    if (!Number.isFinite(nEmbd) || nEmbd < 16 || nEmbd % 16 !== 0)
      return "n_embd must be a multiple of 16 (>= 16).";
    if (!Number.isFinite(nHead) || nHead < 1) return "n_head must be at least 1.";
    if (nEmbd % nHead !== 0) return "n_embd must be divisible by n_head.";
    if (!Number.isFinite(nKvHead) || nKvHead < 1) return "KV heads must be at least 1.";
    if (nHead % nKvHead !== 0) return "n_head must be divisible by the number of KV heads (GQA).";
    if (!Number.isFinite(nLayer) || nLayer < 1) return "n_layer must be at least 1.";
    if (!Number.isFinite(nCtx) || nCtx < 16) return "n_ctx must be at least 16.";
    const window = Number(f.slidingWindow) || 0;
    if (window > nCtx) return "The sliding window cannot exceed n_ctx.";
    const mla = Number(f.mlaLatentDim) || 0;
    if (mla > 0 && mla > nEmbd / nHead) return "The MLA latent cannot exceed the attention head size.";
    if (f.kvEviction === "h2o") {
      if (window > 0) return "Sliding window and KV eviction are mutually exclusive.";
      const budget = Number(f.kvBudget) || 0;
      const sinks = Number(f.sinkTokens) || 0;
      const recent = Number(f.kvRecent) || 0;
      if (budget <= sinks + recent) {
        return "The KV budget must exceed sink tokens + recent tokens.";
      }
    }
  }
  if (step === 2) {
    if (!(Number(f.lr) > 0)) return "Learning rate must be greater than 0.";
    if (!(Number(f.batch) >= 1)) return "Batch size must be at least 1.";
    if (!(Number(f.epochs) >= 1)) return "Epochs must be at least 1.";
    const val = Number(f.valSplit);
    if (!(val >= 0 && val < 1)) return "Validation split must be in [0, 1).";
    if (!(Number(f.gradAccum) >= 1)) return "Gradient accumulation must be at least 1.";
  }
  return null;
}

function buildTrainBody(
  form: FormState,
  datasets: DatasetInfo[],
  resumeOn: boolean,
  resumePath: string,
): { body?: TrainRequestBody; error?: string } {
  const file = form.file || datasets[0]?.name || "";
  if (!file) return { error: "Select a dataset to train." };
  if (resumeOn && !resumePath) {
    return { error: "Select a checkpoint to resume or uncheck the box." };
  }
  const num = (v: string) => Number(v);
  return {
    body: {
      file,
      device: form.device,
      dtype: form.dtype,
      earlyStopPatience: num(form.earlyStop),
      epochs: num(form.epochs),
      batchSize: num(form.batch),
      lr: num(form.lr),
      maxTokens: num(form.maxTokens) || null,
      nCtx: num(form.nCtx),
      nEmbd: num(form.nEmbd),
      nHead: num(form.nHead),
      nLayer: num(form.nLayer),
      valSplit: num(form.valSplit),
      gradAccum: num(form.gradAccum),
      nKvHead: num(form.nKvHead) || null,
      dropout: num(form.dropout),
      attnSoftcap: num(form.attnSoftcap),
      slidingWindow: Number(form.slidingWindow) || 0,
      sinkTokens: Number(form.sinkTokens) || 0,
      kvShareGroup: Number(form.kvShareGroup) || 1,
      mlaLatentDim: Number(form.mlaLatentDim) || 0,
      kvEviction: form.kvEviction,
      kvBudget: Number(form.kvBudget) || 0,
      kvRecent: Number(form.kvRecent) || 64,
      seqWarmupSteps: num(form.seqWarmup),
      weightDecay: num(form.weightDecay),
      labelSmoothing: num(form.labelSmoothing),
      zLoss: num(form.zLoss),
      saveDir: form.saveDir,
      outputDir: form.outputDir,
      resume: resumeOn && resumePath ? resumePath : undefined,
    },
  };
}

function checkpointPlaceholder(
  resumeOn: boolean,
  checkpoints: CheckpointInfo[] | null,
  saveDir: string,
): string {
  if (resumeOn && checkpoints === null) return "loading checkpoints…";
  if (resumeOn && checkpoints !== null && checkpoints.length === 0) {
    return `no checkpoints in ${saveDir || "checkpoints_web"}`;
  }
  return "— select checkpoint —";
}

export function TrainForm() {
  const { job } = useAppState();

  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [resumeOn, setResumeOn] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[] | null>(null);
  const [resumePath, setResumePath] = useState("");
  const [step, setStep] = useState(0);
  const [notice, setNotice] = useState<{ text: string; tone: NoticeTone }>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);
  const epochsSynced = useRef(false);

  const set = (key: keyof FormState, value: string) => setForm((f) => ({ ...f, [key]: value }));

  const applyPreset = (name: string) => {
    setForm((f) => {
      const preset = PRESETS[name];
      return preset ? { ...f, preset: name, ...preset } : { ...f, preset: "custom" };
    });
  };

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([api.getDatasets(), api.getDevices()]).then(([ds, dv]) => {
      if (cancelled) return;
      // Biggest corpora first: the default selection should not be a tiny
      // test fixture.
      if (ds.status === "fulfilled") {
        setDatasets([...ds.value].sort((a, b) => b.size_bytes - a.size_bytes));
      }
      if (dv.status === "fulfilled") setDevices(dv.value);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Reflect the running job's epoch count once per run, so the user can still
  // edit the field for the next run while one is active.
  useEffect(() => {
    if (job.status === "running" && job.epochs > 0 && !epochsSynced.current) {
      epochsSynced.current = true;
      setForm((f) => ({ ...f, epochs: String(job.epochs) }));
    } else if (job.status !== "running") {
      epochsSynced.current = false;
    }
  }, [job.status, job.epochs]);

  const loadCheckpoints = useCallback(async (dir: string) => {
    setCheckpoints(null);
    setResumePath("");
    try {
      const list = await api.getCheckpoints(dir || "checkpoints_web");
      setCheckpoints(list);
    } catch (e) {
      setCheckpoints([]);
      setNotice({
        text: e instanceof Error ? `Failed to list checkpoints: ${e.message}` : "Failed to list checkpoints",
        tone: "error",
      });
    }
  }, []);

  const toggleResume = (enabled: boolean) => {
    setResumeOn(enabled);
    if (enabled) void loadCheckpoints(form.saveDir);
    else setResumePath("");
  };

  const onSaveDirChange = (value: string) => {
    set("saveDir", value);
    if (resumeOn) void loadCheckpoints(value);
  };

  const next = () => {
    const error = validateStep(step, form, datasets);
    if (error) {
      setNotice({ text: error, tone: "error" });
      return;
    }
    setNotice({ text: "", tone: "muted" });
    setStep((s) => Math.min(s + 1, STEPS.length - 1));
  };

  const start = async () => {
    if (busy) return;
    for (let s = 0; s < STEPS.length; s += 1) {
      const error = validateStep(s, form, datasets);
      if (error) {
        setStep(s);
        setNotice({ text: error, tone: "error" });
        return;
      }
    }
    const { body, error } = buildTrainBody(form, datasets, resumeOn, resumePath);
    if (!body) {
      setNotice({ text: error ?? "Invalid configuration.", tone: "error" });
      return;
    }
    setBusy(true);
    setNotice({ text: "Starting…", tone: "muted" });
    try {
      const res = await api.startTrain(body);
      setNotice({
        text: res.jobId ? `Training started (${res.jobId})` : "Training started",
        tone: "ok",
      });
      setResumeOn(false);
      setResumePath("");
      setCheckpoints(null);
    } catch (e) {
      setNotice({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  const running = job.status === "running";

  return (
    <div className="glass-card rounded-2xl p-5 self-start min-w-0">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">
        Training wizard
      </h3>

      <ol className="mb-5 grid grid-cols-4 gap-1">
        {STEPS.map((label, i) => (
          <li key={label} className="min-w-0">
            <button
              type="button"
              onClick={() => setStep(i)}
              className={`w-full min-w-0 truncate rounded-lg px-1 py-1.5 text-[10px] leading-tight transition ${
                i === step
                  ? "bg-violet-500/20 text-violet-200"
                  : i < step
                    ? "text-emerald-300 hover:bg-slate-800/60"
                    : "text-slate-500 hover:bg-slate-800/60"
              }`}
            >
              {i + 1}. {label}
            </button>
          </li>
        ))}
      </ol>

      {step === 0 ? (
        <StepData
          form={form}
          set={set}
          datasets={datasets}
          devices={devices}
          onSaveDirChange={onSaveDirChange}
          resumeOn={resumeOn}
          checkpoints={checkpoints}
          resumePath={resumePath}
          onToggleResume={toggleResume}
          onSelectResume={setResumePath}
        />
      ) : null}
      {step === 1 ? <StepArchitecture form={form} set={set} onPreset={applyPreset} /> : null}
      {step === 2 ? <StepTraining form={form} set={set} /> : null}
      {step === 3 ? (
        <StepReview form={form} datasets={datasets} resumeOn={resumeOn} resumePath={resumePath} />
      ) : null}

      <div className="mt-5 flex gap-2">
        {step > 0 ? (
          <Button variant="ghost" className="flex-1" onClick={() => setStep((s) => s - 1)}>
            Back
          </Button>
        ) : null}
        {step < STEPS.length - 1 ? (
          <Button variant="primary" className="flex-1" onClick={next}>
            Next
          </Button>
        ) : (
          <Button variant="primary" className="flex-1" disabled={busy || running} onClick={() => void start()}>
            {busy ? "Starting…" : "Start training"}
          </Button>
        )}
      </div>

      <p className={`text-xs mt-3 ${NOTICE_CLASSES[notice.tone]}`}>{notice.text}</p>
    </div>
  );
}

interface StepDataProps {
  form: FormState;
  set: (key: keyof FormState, value: string) => void;
  datasets: DatasetInfo[];
  devices: DeviceInfo[];
  onSaveDirChange: (value: string) => void;
  resumeOn: boolean;
  checkpoints: CheckpointInfo[] | null;
  resumePath: string;
  onToggleResume: (enabled: boolean) => void;
  onSelectResume: (path: string) => void;
}

function StepData({
  form,
  set,
  datasets,
  devices,
  onSaveDirChange,
  resumeOn,
  checkpoints,
  resumePath,
  onToggleResume,
  onSelectResume,
}: StepDataProps) {
  const selectedDataset = form.file || datasets[0]?.name || "";
  const resumeDisabled = !resumeOn || checkpoints === null || checkpoints.length === 0;
  return (
    <>
      <Field label="Dataset">
        <Select value={selectedDataset} onChange={(e) => set("file", e.target.value)}>
          {datasets.length === 0 ? (
            <option value="">no datasets</option>
          ) : (
            datasets.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
              </option>
            ))
          )}
        </Select>
      </Field>
      <p className="text-[11px] text-slate-500 mt-1 mb-3">
        Datasets are tokenized with the working-directory tokenizer; 1 epoch over 10M tokens is a good default.
      </p>

      <Field label="Device">
        <Select value={form.device} onChange={(e) => set("device", e.target.value)}>
          {devices.length === 0 ? (
            <option value="DefaultDevice">DefaultDevice</option>
          ) : (
            devices.map((d) => {
              const selector = d.selector || d.name;
              return (
                <option key={selector} value={selector}>
                  {selector}
                </option>
              );
            })
          )}
        </Select>
      </Field>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <Field label="Save dir (checkpoints)">
          <Input type="text" value={form.saveDir} onChange={(e) => onSaveDirChange(e.target.value)} />
        </Field>
        <Field label="Output dir (model)">
          <Input type="text" value={form.outputDir} onChange={(e) => set("outputDir", e.target.value)} />
        </Field>
      </div>

      <div className="mt-4">
        <label className="flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={resumeOn}
            onChange={(e) => onToggleResume(e.target.checked)}
            className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
          />
          Resume weights from a checkpoint (warm start)
        </label>
        <div className="mt-1.5">
          <Select value={resumePath} onChange={(e) => onSelectResume(e.target.value)} disabled={resumeDisabled}>
            <option value="">{checkpointPlaceholder(resumeOn, checkpoints, form.saveDir)}</option>
            {(checkpoints ?? []).map((cp) => (
              <option key={cp.path} value={cp.path}>
                {cp.name} · {fmtBytes(cp.size_bytes)}
              </option>
            ))}
          </Select>
        </div>
      </div>
    </>
  );
}

function StepArchitecture({
  form,
  set,
  onPreset,
}: {
  form: FormState;
  set: (key: keyof FormState, value: string) => void;
  onPreset: (name: string) => void;
}) {
  const custom = form.preset === "custom";
  const params = paramCount(form);
  return (
    <>
      <Field label="Preset">
        <Select value={form.preset} onChange={(e) => onPreset(e.target.value)}>
          <option value="nano">nano · ~2M params</option>
          <option value="tiny">tiny · ~5.5M params</option>
          <option value="small">small · ~30M params</option>
          <option value="base">base · ~100M params</option>
          <option value="custom">custom · set dimensions manually</option>
        </Select>
      </Field>
      <p className="text-[11px] text-slate-500 mt-1 mb-3">
        Presets mirror the CLI. Estimated size:{" "}
        <span className="text-slate-300 tnum font-mono">{(params / 1e6).toFixed(2)}M params</span>
        {" · "}
        <span className="text-slate-300 tnum font-mono">{fmtBytes(params * 4)}</span> f32 weights.
      </p>

      <div className="grid grid-cols-2 gap-3">
        <NumberField label="n_embd" min={16} value={form.nEmbd} onChange={(v) => set("nEmbd", v)} disabled={!custom} />
        <NumberField label="n_head" min={1} value={form.nHead} onChange={(v) => set("nHead", v)} disabled={!custom} />
        <NumberField label="KV heads (GQA)" min={1} value={form.nKvHead} onChange={(v) => set("nKvHead", v)} disabled={!custom} />
        <NumberField label="n_layer" min={1} value={form.nLayer} onChange={(v) => set("nLayer", v)} disabled={!custom} />
        <NumberField label="n_ctx" min={16} value={form.nCtx} onChange={(v) => set("nCtx", v)} disabled={!custom} />
        <NumberField label="Dropout" step={0.05} min={0} max={0.9} value={form.dropout} onChange={(v) => set("dropout", v)} />
      </div>

      <details className="mt-4 rounded-xl border border-slate-800/70 bg-slate-950/30 px-3 py-2">
        <summary className="text-xs text-slate-400 cursor-pointer select-none">
          Attention & cache (advanced)
        </summary>
        <div className="grid grid-cols-2 gap-3 mt-3">
          <NumberField label="Sliding window" step={1} min={0} value={form.slidingWindow} onChange={(v) => set("slidingWindow", v)} />
          <NumberField label="Sink tokens" step={1} min={0} value={form.sinkTokens} onChange={(v) => set("sinkTokens", v)} />
          <NumberField label="KV share group" step={1} min={1} value={form.kvShareGroup} onChange={(v) => set("kvShareGroup", v)} />
          <NumberField label="MLA latent dim" step={16} min={0} value={form.mlaLatentDim} onChange={(v) => set("mlaLatentDim", v)} />
          <NumberField label="Attn softcap" step={1} min={0} value={form.attnSoftcap} onChange={(v) => set("attnSoftcap", v)} />
        </div>
        <div className="grid grid-cols-3 gap-3 mt-3">
          <Field label="KV eviction">
            <Select value={form.kvEviction} onChange={(e) => set("kvEviction", e.target.value)}>
              <option value="none">none</option>
              <option value="h2o">h2o · heavy hitters</option>
            </Select>
          </Field>
          <NumberField label="KV budget (rows)" step={64} min={0} value={form.kvBudget} onChange={(v) => set("kvBudget", v)} />
          <NumberField label="KV recent tokens" step={16} min={0} value={form.kvRecent} onChange={(v) => set("kvRecent", v)} />
        </div>
        <p className="text-[10px] text-slate-500 mt-2">
          Sliding window + sinks bound the KV cache. KV share group reuses K/V projections across layers.
          MLA compresses K/V into a latent (mutually exclusive with GQA in practice: set KV heads = heads
          when using MLA). H2O eviction trains the attention mask for a bounded heavy-hitter cache
          (mutually exclusive with the sliding window; it needs a budget above sink + recent tokens).
        </p>
      </details>
    </>
  );
}

function StepTraining({
  form,
  set,
}: {
  form: FormState;
  set: (key: keyof FormState, value: string) => void;
}) {
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Epochs" min={1} value={form.epochs} onChange={(v) => set("epochs", v)} />
        <NumberField label="Batch" min={1} value={form.batch} onChange={(v) => set("batch", v)} />
        <NumberField label="LR" step={0.00001} min={0} value={form.lr} onChange={(v) => set("lr", v)} />
        <NumberField label="Max tokens" min={0} value={form.maxTokens} onChange={(v) => set("maxTokens", v)} />
        <NumberField label="Val split" step={0.01} min={0} max={0.9} value={form.valSplit} onChange={(v) => set("valSplit", v)} />
        <NumberField label="Grad accum" min={1} value={form.gradAccum} onChange={(v) => set("gradAccum", v)} />
        <NumberField label="Seq warmup steps" min={0} value={form.seqWarmup} onChange={(v) => set("seqWarmup", v)} />
        <NumberField label="Early stop (epochs)" min={0} value={form.earlyStop} onChange={(v) => set("earlyStop", v)} />
      </div>
      <details className="mt-4 rounded-xl border border-slate-800/70 bg-slate-950/30 px-3 py-2">
        <summary className="text-xs text-slate-400 cursor-pointer select-none">
          Regularization (advanced)
        </summary>
        <div className="grid grid-cols-2 gap-3 mt-3">
          <NumberField label="Weight decay" step={0.01} min={0} value={form.weightDecay} onChange={(v) => set("weightDecay", v)} />
          <NumberField label="Label smoothing" step={0.01} min={0} max={0.9} value={form.labelSmoothing} onChange={(v) => set("labelSmoothing", v)} />
          <NumberField label="Z-loss" step={0.0001} min={0} value={form.zLoss} onChange={(v) => set("zLoss", v)} />
        </div>
      </details>
    </>
  );
}

function StepReview({
  form,
  datasets,
  resumeOn,
  resumePath,
}: {
  form: FormState;
  datasets: DatasetInfo[];
  resumeOn: boolean;
  resumePath: string;
}) {
  const file = form.file || datasets[0]?.name || "—";
  const params = paramCount(form);
  const rows: [string, string][] = [
    ["Dataset", file],
    ["Device", form.device],
    ["Preset", form.preset],
    ["Architecture", `d ${form.nEmbd} · heads ${form.nHead} · kv ${form.nKvHead} · layers ${form.nLayer} · ctx ${form.nCtx}`],
    [
      "KV cache",
      form.kvEviction === "h2o"
        ? `h2o eviction · budget ${form.kvBudget} · recent ${form.kvRecent}`
        : Number(form.slidingWindow) > 0
          ? `sliding window ${form.slidingWindow} · sinks ${form.sinkTokens}`
          : "dense",
    ],
    ["Params (f32)", `${(params / 1e6).toFixed(2)}M · ${fmtBytes(params * 4)}`],
    ["Schedule", `${form.epochs} epoch(s) · batch ${form.batch} · lr ${form.lr} · accum ${form.gradAccum}`],
    ["Token budget", Number(form.maxTokens) > 0 ? Number(form.maxTokens).toLocaleString() : "full dataset"],
    ["Validation split", form.valSplit],
    ["Precision", "f32 (f16 training is not supported)"],
    ["Output", form.outputDir],
    ["Checkpoints", form.saveDir],
    ["Resume", resumeOn && resumePath ? resumePath : "no"],
  ];
  return (
    <div className="rounded-xl border border-slate-800/70 bg-slate-950/30 divide-y divide-slate-800/60">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-3 px-3 py-2 text-xs">
          <span className="text-slate-500">{label}</span>
          <span className="text-slate-300 font-mono text-right truncate">{value}</span>
        </div>
      ))}
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </Field>
  );
}
