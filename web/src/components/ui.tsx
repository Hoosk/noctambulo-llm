import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { Icon, type IconName } from "./Icon";

const TONES = {
  slate: "bg-slate-800/70 text-slate-300",
  violet: "bg-amber-500/15 text-amber-300",
  fuchsia: "bg-orange-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
} as const;

export type Tone = keyof typeof TONES;

const BORDERS: Record<Tone, string> = {
  slate: "border-slate-700",
  violet: "border-amber-500/30",
  fuchsia: "border-fuchsia-500/30",
  emerald: "border-emerald-500/30",
  amber: "border-amber-500/30",
  rose: "border-rose-500/30",
  sky: "border-sky-500/30",
};

export function ViewHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-7">
      <div className="flex items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="mb-2 flex items-center gap-2.5">
            <span className="h-px w-7 bg-amber-500/70" />
            <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-amber-300/70">
              Instrument
            </span>
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-medium text-ivory tracking-tight">
            {title}
          </h2>
          {subtitle ? (
            <p className="text-sm text-slate-400 mt-1.5 max-w-2xl">{subtitle}</p>
          ) : null}
        </div>
        {actions ? <div className="flex gap-2 shrink-0">{actions}</div> : null}
      </div>
      <div className="mt-4 h-px bg-gradient-to-r from-brass/60 via-slate-700/40 to-transparent" />
    </div>
  );
}

export function StatCard({
  label,
  value,
  icon,
  tone = "slate",
  sub,
  valueClass = "text-ivory",
}: {
  label: string;
  value: ReactNode;
  icon?: IconName;
  tone?: Tone;
  sub?: ReactNode;
  valueClass?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-lg border border-slate-800/80 bg-slate-950/55 px-3.5 py-3 min-w-0 transition-colors hover:border-brass/50">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[9px] font-medium uppercase tracking-[0.16em] text-slate-500 truncate">
          {label}
        </span>
        {icon ? (
          <div className={`w-7 h-7 rounded-md flex items-center justify-center ${TONES[tone]}`}>
            <Icon name={icon} className="w-3.5 h-3.5" />
          </div>
        ) : null}
      </div>
      <div className={`tnum font-mono text-lg font-semibold mt-1.5 truncate ${valueClass}`}>
        {value}
      </div>
      {sub ? <div className="text-[10px] text-slate-500 mt-0.5 truncate">{sub}</div> : null}
      <span className="pointer-events-none absolute bottom-0 left-3 right-3 h-px bg-gradient-to-r from-transparent via-amber-500/20 to-transparent" />
    </div>
  );
}

export function Badge({
  children,
  tone = "slate",
  pulse = false,
}: {
  children: ReactNode;
  tone?: Tone;
  pulse?: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono text-[9px] font-medium uppercase tracking-[0.12em] px-1.5 py-0.5 rounded-[3px] border ${TONES[tone]} ${BORDERS[tone]}`}
    >
      {pulse ? (
        <span className="relative flex w-1.5 h-1.5">
          <span className="absolute inline-flex w-full h-full rounded-full opacity-60 animate-ping bg-current" />
          <span className="relative inline-flex w-1.5 h-1.5 rounded-full bg-current" />
        </span>
      ) : null}
      {children}
    </span>
  );
}

const FIELD_BASE =
  "w-full px-3 py-2 rounded-md bg-slate-950/75 border border-slate-700/70 text-ivory text-sm placeholder-slate-600 focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 transition";

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="block font-mono text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-500 mb-1.5">
        {label}
      </span>
      {children}
    </label>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  const { className = "", ...rest } = props;
  return <input className={`${FIELD_BASE} ${className}`} {...rest} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  const { className = "", ...rest } = props;
  return <select className={`${FIELD_BASE} ${className}`} {...rest} />;
}

export function Textarea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const { className = "", ...rest } = props;
  return <textarea className={`${FIELD_BASE} resize-y leading-relaxed ${className}`} {...rest} />;
}
