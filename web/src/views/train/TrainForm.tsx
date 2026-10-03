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
  seqWarmup: string;
  weightDecay: string;
  labelSmoothing: string;
  zLoss: string;
  saveDir: string;
  outputDir: string;
}

const DEFAULT_FORM: FormState = {
  file: "",
  device: "DefaultDevice",
  dtype: "f32",
  earlyStop: "0",
  epochs: "1",
  batch: "32",
  lr: "0.00005",
  maxTokens: "10000000",
  nCtx: "256",
  nEmbd: "256",
  nHead: "4",
  nLayer: "4",
  valSplit: "0.02",
  gradAccum: "1",
  nKvHead: "0",
  dropout: "0.1",
  attnSoftcap: "0",
  slidingWindow: "0",
  sinkTokens: "0",
  kvShareGroup: "1",
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
  const [notice, setNotice] = useState<{ text: string; tone: NoticeTone }>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);
  const epochsSynced = useRef(false);

  const set = (key: keyof FormState, value: string) => setForm((f) => ({ ...f, [key]: value }));

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([api.getDatasets(), api.getDevices()]).then(([ds, dv]) => {
      if (cancelled) return;
      if (ds.status === "fulfilled") setDatasets(ds.value);
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

  const start = async () => {
    if (busy) return;
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

  const action = async (run: () => Promise<unknown>, done: string, tone: NoticeTone) => {
    try {
      await run();
      setNotice({ text: done, tone });
    } catch (e) {
      setNotice({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
    }
  };

  const running = job.status === "running";

  return (
    <div className="glass-card rounded-2xl p-5 lg:sticky lg:top-6 self-start min-w-0">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">
        Configuration
      </h3>
      <ConfigFields
        form={form}
        set={set}
        datasets={datasets}
        devices={devices}
        onSaveDirChange={onSaveDirChange}
      />
      <ResumePicker
        resumeOn={resumeOn}
        checkpoints={checkpoints}
        resumePath={resumePath}
        saveDir={form.saveDir}
        onToggle={toggleResume}
        onSelect={setResumePath}
      />
      <TrainActions
        running={running}
        paused={job.paused}
        stopping={job.stopping}
        busy={busy}
        onStart={() => void start()}
        onTogglePause={() =>
          void (job.paused
            ? action(api.resumeTrain, "Resuming…", "ok")
            : action(api.pauseTrain, "Pausing…", "warn"))
        }
        onStop={() => void action(api.stopTrain, "Stopping…", "warn")}
      />
      <p className={`text-xs mt-3 ${NOTICE_CLASSES[notice.tone]}`}>{notice.text}</p>
    </div>
  );
}

interface ConfigFieldsProps {
  form: FormState;
  set: (key: keyof FormState, value: string) => void;
  datasets: DatasetInfo[];
  devices: DeviceInfo[];
  onSaveDirChange: (value: string) => void;
}

function ConfigFields({ form, set, datasets, devices, onSaveDirChange }: ConfigFieldsProps) {
  const selectedDataset = form.file || datasets[0]?.name || "";
  return (
    <>
      <div className="mb-3">
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
      </div>

      <div className="mb-3">
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
      </div>

      <div className="mb-4">
        <Field label="Compute precision (dtype)">
          <Select value={form.dtype} onChange={(e) => set("dtype", e.target.value)}>
            <option value="f32">f32 · full precision</option>
            <option value="f16">f16 · faster on supported GPUs</option>
          </Select>
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Epochs" min={1} value={form.epochs} onChange={(v) => set("epochs", v)} />
        <NumberField label="Batch" min={1} value={form.batch} onChange={(v) => set("batch", v)} />
        <NumberField label="LR" step={0.00001} min={0} value={form.lr} onChange={(v) => set("lr", v)} />
        <NumberField label="Max tokens" min={0} value={form.maxTokens} onChange={(v) => set("maxTokens", v)} />
        <NumberField label="n_ctx" min={16} value={form.nCtx} onChange={(v) => set("nCtx", v)} />
        <NumberField label="n_embd" min={16} value={form.nEmbd} onChange={(v) => set("nEmbd", v)} />
        <NumberField label="n_head" min={1} value={form.nHead} onChange={(v) => set("nHead", v)} />
        <NumberField label="n_layer" min={1} value={form.nLayer} onChange={(v) => set("nLayer", v)} />
        <NumberField label="Val split" step={0.01} min={0} max={0.9} value={form.valSplit} onChange={(v) => set("valSplit", v)} />
        <NumberField label="Grad accum" min={1} value={form.gradAccum} onChange={(v) => set("gradAccum", v)} />
        <NumberField label="KV heads (GQA)" min={0} value={form.nKvHead} onChange={(v) => set("nKvHead", v)} />
        <NumberField label="Dropout" step={0.05} min={0} max={0.9} value={form.dropout} onChange={(v) => set("dropout", v)} />
        <NumberField label="Attn softcap" step={1} min={0} value={form.attnSoftcap} onChange={(v) => set("attnSoftcap", v)} />
        <NumberField label="Sliding window" step={1} min={0} value={form.slidingWindow} onChange={(v) => set("slidingWindow", v)} />
        <NumberField label="Sink tokens" step={1} min={0} value={form.sinkTokens} onChange={(v) => set("sinkTokens", v)} />
        <NumberField label="KV share group" step={1} min={1} value={form.kvShareGroup} onChange={(v) => set("kvShareGroup", v)} />
        <NumberField label="Seq warmup steps" min={0} value={form.seqWarmup} onChange={(v) => set("seqWarmup", v)} />
        <NumberField label="Early stop (epochs)" min={0} value={form.earlyStop} onChange={(v) => set("earlyStop", v)} />
        <NumberField label="Weight decay" step={0.01} min={0} value={form.weightDecay} onChange={(v) => set("weightDecay", v)} />
        <NumberField label="Label smoothing" step={0.01} min={0} max={0.9} value={form.labelSmoothing} onChange={(v) => set("labelSmoothing", v)} />
        <NumberField label="Z-loss" step={0.0001} min={0} value={form.zLoss} onChange={(v) => set("zLoss", v)} />
      </div>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <Field label="Save dir">
          <Input type="text" value={form.saveDir} onChange={(e) => onSaveDirChange(e.target.value)} />
        </Field>
        <Field label="Output dir">
          <Input type="text" value={form.outputDir} onChange={(e) => set("outputDir", e.target.value)} />
        </Field>
      </div>
    </>
  );
}

interface ResumePickerProps {
  resumeOn: boolean;
  checkpoints: CheckpointInfo[] | null;
  resumePath: string;
  saveDir: string;
  onToggle: (enabled: boolean) => void;
  onSelect: (path: string) => void;
}

function ResumePicker({ resumeOn, checkpoints, resumePath, saveDir, onToggle, onSelect }: ResumePickerProps) {
  const disabled = !resumeOn || checkpoints === null || checkpoints.length === 0;
  return (
    <div className="mt-4">
      <label className="flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={resumeOn}
          onChange={(e) => onToggle(e.target.checked)}
          className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
        />
        Resume weights from checkpoint (warm start)
      </label>
      <div className="mt-1.5">
        <Select value={resumePath} onChange={(e) => onSelect(e.target.value)} disabled={disabled}>
          <option value="">{checkpointPlaceholder(resumeOn, checkpoints, saveDir)}</option>
          {(checkpoints ?? []).map((cp) => (
            <option key={cp.path} value={cp.path}>
              {cp.name} · {fmtBytes(cp.size_bytes)}
            </option>
          ))}
        </Select>
      </div>
    </div>
  );
}

interface TrainActionsProps {
  running: boolean;
  paused: boolean;
  stopping: boolean;
  busy: boolean;
  onStart: () => void;
  onTogglePause: () => void;
  onStop: () => void;
}

function TrainActions({ running, paused, stopping, busy, onStart, onTogglePause, onStop }: TrainActionsProps) {
  return (
    <div className="flex gap-3 mt-5">
      <Button variant="primary" className="flex-1" disabled={running || busy} onClick={onStart}>
        {busy ? "Starting…" : "Start"}
      </Button>
      <Button variant={paused ? "success" : "warn"} className="flex-1" disabled={!running || stopping} onClick={onTogglePause}>
        {paused ? "Resume" : "Pause"}
      </Button>
      <Button variant="danger" className="flex-1" disabled={!running || stopping} onClick={onStop}>
        {stopping ? "Stopping…" : "Stop"}
      </Button>
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
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <Field label={label}>
      <Input type="number" min={min} max={max} step={step} value={value} onChange={(e) => onChange(e.target.value)} />
    </Field>
  );
}
