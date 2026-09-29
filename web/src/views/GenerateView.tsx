import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ModelInfo } from "../api/types";
import { ViewHeader } from "../components/ui";
import { readSseData } from "../lib/sseStream";
import { useAppState } from "../state/appContext";
import { GenerateForm, type GenForm } from "./generate/GenerateForm";
import { GenerateOutput, type StatusTone } from "./generate/GenerateOutput";

const DEFAULT_FORM: GenForm = {
  prompt: "Once upon a time",
  maxTokens: "60",
  temperature: "0.8",
  topK: "40",
  topP: "0.9",
  minP: "0.05",
  rep: "1.2",
  ropeScaling: "none",
  ropeFactor: "1",
};

interface RunStatus {
  text: string;
  tone: StatusTone;
}

export function GenerateView() {
  const { activeModel } = useAppState();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelDir, setModelDir] = useState(activeModel);
  const [form, setForm] = useState<GenForm>(DEFAULT_FORM);
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<RunStatus>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getModels()
      .then((list) => {
        if (!cancelled) setModels(list);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setStatus({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectedDir = modelDir || models.find((m) => m.default)?.dir || models[0]?.dir || "";
  const onField = (key: keyof GenForm, value: string) => setForm((f) => ({ ...f, [key]: value }));

  const run = async () => {
    if (busy || !selectedDir) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setOutput("");
    setStatus({ text: "", tone: "muted" });
    const started = performance.now();
    let text = "";
    let tokens = 0;

    const onData = (payload: string) => {
      try {
        const ev = JSON.parse(payload) as { token?: string };
        if (!ev.token) return;
        text += ev.token;
        tokens += 1;
        setOutput(text);
        const secs = Math.max((performance.now() - started) / 1000, 0.001);
        setStatus({ text: `${tokens} tokens · ${(tokens / secs).toFixed(1)} tok/s`, tone: "running" });
      } catch {
        /* keep-alive or malformed chunk */
      }
    };

    try {
      const res = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelDir: selectedDir,
          prompt: form.prompt,
          maxTokens: Number(form.maxTokens),
          temperature: Number(form.temperature),
          topK: Number(form.topK),
          topP: Number(form.topP),
          minP: Number(form.minP),
          repetitionPenalty: Number(form.rep),
          ropeScaling: form.ropeScaling,
          ropeFactor: Number(form.ropeFactor) || 1,
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = (await res.json()) as { error?: string };
          if (body.error) detail = body.error;
        } catch {
          /* not json */
        }
        throw new Error(detail);
      }
      await readSseData(res, onData);
      const secs = Math.max((performance.now() - started) / 1000, 0.001);
      setStatus({
        text: `Done: ${tokens} tokens in ${secs.toFixed(2)}s · ${(tokens / secs).toFixed(1)} tok/s`,
        tone: "ok",
      });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setStatus({ text: `Generation stopped (${tokens} tokens).`, tone: "warn" });
      } else {
        setStatus({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  };

  return (
    <>
      <ViewHeader title="Generate" subtitle="Autoregressive generation with sampling parameters." />
      {models.length === 0 ? (
        <div className="mb-5 p-4 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500">
          No models yet. Train one first.
        </div>
      ) : null}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
        <GenerateForm
          models={models}
          modelDir={selectedDir}
          onModelChange={setModelDir}
          form={form}
          onField={onField}
          busy={busy}
          onRun={() => void run()}
          onStop={() => abortRef.current?.abort()}
        />
        <GenerateOutput text={output} status={status} />
      </div>
    </>
  );
}
