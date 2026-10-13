import type { ModelInfo } from "../../api/types";
import { Button } from "../../components/Button";
import { Field, Input, Select, Textarea } from "../../components/ui";

export interface GenForm {
  prompt: string;
  maxTokens: string;
  temperature: string;
  topK: string;
  topP: string;
  minP: string;
  rep: string;
  ropeScaling: string;
  ropeFactor: string;
  dtype: string;
  kvCacheDtype: string;
  kvEviction: string;
  kvBudget: string;
  kvRecent: string;
  kvSpill: boolean;
  kvMaxBudget: string;
  kvGrow: string;
  kvGrowInterval: string;
}

interface Props {
  models: ModelInfo[];
  modelDir: string;
  onModelChange: (dir: string) => void;
  form: GenForm;
  onField: (key: keyof GenForm, value: string | boolean) => void;
  busy: boolean;
  onRun: () => void;
  onStop: () => void;
}

export function GenerateForm({
  models,
  modelDir,
  onModelChange,
  form,
  onField,
  busy,
  onRun,
  onStop,
}: Props) {
  return (
    <div className="glass-card rounded-2xl p-5 min-w-0">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">
        Configuration
      </h3>

      <div className="mb-3">
        <Field label="Model">
          <Select value={modelDir} onChange={(e) => onModelChange(e.target.value)}>
            {models.length === 0 ? (
              <option value="">no models</option>
            ) : (
              models.map((m) => (
                <option key={m.dir} value={m.dir} disabled={!m.compatible}>
                  {m.name} · {m.params_m.toFixed(1)}M
                  {m.compatible ? "" : " · incompatible (older architecture)"}
                </option>
              ))
            )}
          </Select>
        </Field>
      </div>

      <div className="mb-3">
        <Field label="Prompt">
          <Textarea
            rows={4}
            placeholder="Once upon a time..."
            value={form.prompt}
            onChange={(e) => onField("prompt", e.target.value)}
          />
        </Field>
      </div>

      <div className="grid grid-cols-3 gap-3 mt-2">
        <Field label="Max tokens">
          <Input type="number" min={1} step={1} value={form.maxTokens} onChange={(e) => onField("maxTokens", e.target.value)} />
        </Field>
        <Field label="Temperature">
          <Input type="number" min={0} step={0.05} value={form.temperature} onChange={(e) => onField("temperature", e.target.value)} />
        </Field>
        <Field label="Top-K">
          <Input type="number" min={1} step={1} value={form.topK} onChange={(e) => onField("topK", e.target.value)} />
        </Field>
      </div>
      <div className="grid grid-cols-3 gap-3 mt-3">
        <Field label="Top-P">
          <Input type="number" min={0} step={0.05} value={form.topP} onChange={(e) => onField("topP", e.target.value)} />
        </Field>
        <Field label="Min-P">
          <Input type="number" min={0} step={0.01} value={form.minP} onChange={(e) => onField("minP", e.target.value)} />
        </Field>
        <Field label="Repetition">
          <Input type="number" min={1} step={0.05} value={form.rep} onChange={(e) => onField("rep", e.target.value)} />
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <Field label="RoPE scaling">
          <Select value={form.ropeScaling} onChange={(e) => onField("ropeScaling", e.target.value)}>
            <option value="none">none</option>
            <option value="linear">linear</option>
            <option value="ntk">ntk (dynamic)</option>
            <option value="yarn">yarn</option>
          </Select>
        </Field>
        <Field label="RoPE factor">
          <Input
            type="number"
            min={1}
            step={0.5}
            value={form.ropeFactor}
            onChange={(e) => onField("ropeFactor", e.target.value)}
          />
        </Field>
      </div>
      <p className="text-[11px] text-slate-500 -mt-1 mb-1">
        Scaling extends the usable context past the trained window (generation past n_ctx works with or without it).
      </p>

      <div className="grid grid-cols-2 gap-3 mt-3">
        <Field label="Compute precision">
          <Select value={form.dtype} onChange={(e) => onField("dtype", e.target.value)}>
            <option value="f32">f32 · full precision</option>
            <option value="f16">f16 · faster on supported GPUs</option>
          </Select>
        </Field>
        <Field label="KV cache precision">
          <Select
            value={form.kvCacheDtype}
            onChange={(e) => onField("kvCacheDtype", e.target.value)}
          >
            <option value="f32">f32 · full precision</option>
            <option value="f16">f16 · half memory</option>
            <option value="int8">int8 · quarter memory</option>
            <option value="q4">q4 · eighth memory</option>
          </Select>
        </Field>
      </div>

      <div className="grid grid-cols-3 gap-3 mt-3">
        <Field label="KV eviction">
          <Select value={form.kvEviction} onChange={(e) => onField("kvEviction", e.target.value)}>
            <option value="">model default</option>
            <option value="h2o">h2o · heavy hitters</option>
            <option value="none">none · dense</option>
          </Select>
        </Field>
        <Field label="Budget (rows)">
          <Input
            type="number"
            min={0}
            step={64}
            placeholder="model default"
            value={form.kvBudget}
            onChange={(e) => onField("kvBudget", e.target.value)}
          />
        </Field>
        <Field label="Recent tokens">
          <Input
            type="number"
            min={0}
            step={16}
            placeholder="model default"
            value={form.kvRecent}
            onChange={(e) => onField("kvRecent", e.target.value)}
          />
        </Field>
      </div>
      <p className="text-[11px] text-slate-500 -mt-1 mb-1">
        Heavy-hitter eviction bounds the cache to the budget, keeping sinks, recent tokens and the rows
        with the highest accumulated attention. Empty/0 keeps the checkpoint config.
      </p>

      <details className="mt-2 rounded-xl border border-slate-800/70 bg-slate-950/30 px-3 py-2">
        <summary className="text-xs text-slate-400 cursor-pointer select-none">
          Spill to host RAM (advanced)
        </summary>
        <label className="flex items-center gap-2.5 text-xs text-slate-400 cursor-pointer select-none mt-3">
          <input
            type="checkbox"
            checked={form.kvSpill}
            onChange={(e) => onField("kvSpill", e.target.checked)}
            className="w-4 h-4 rounded accent-violet-500 cursor-pointer"
          />
          Archive evicted rows in host RAM and replay them
        </label>
        <div className="grid grid-cols-3 gap-3 mt-3">
          <Field label="Max budget (rows)">
            <Input
              type="number"
              min={0}
              step={64}
              placeholder="= budget"
              value={form.kvMaxBudget}
              onChange={(e) => onField("kvMaxBudget", e.target.value)}
            />
          </Field>
          <Field label="Grow rows">
            <Input
              type="number"
              min={0}
              step={32}
              value={form.kvGrow}
              onChange={(e) => onField("kvGrow", e.target.value)}
            />
          </Field>
          <Field label="Grow every (tokens)">
            <Input
              type="number"
              min={1}
              step={16}
              value={form.kvGrowInterval}
              onChange={(e) => onField("kvGrowInterval", e.target.value)}
            />
          </Field>
        </div>
        <p className="text-[10px] text-slate-500 mt-2">
          With spill on, the resident budget can grow up to the max budget, refilling each new slot with
          the highest-score archived rows. Growth needs h2o eviction and a max budget above the initial one.
        </p>
      </details>

      <div className="mt-5 flex gap-3">
        <Button variant="primary" disabled={busy || !modelDir} onClick={onRun}>
          {busy ? "Generating…" : "Generate"}
        </Button>
        <Button variant="danger" disabled={!busy} onClick={onStop}>
          Stop
        </Button>
      </div>
    </div>
  );
}
