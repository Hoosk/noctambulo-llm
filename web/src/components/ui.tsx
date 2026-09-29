import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { Icon, type IconName } from "./Icon";

const TONES = {
  slate: "bg-slate-700/40 text-slate-300",
  violet: "bg-violet-500/15 text-violet-300",
  fuchsia: "bg-fuchsia-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
} as const;

export type Tone = keyof typeof TONES;

const BORDERS: Record<Tone, string> = {
  slate: "border-slate-700",
  violet: "border-violet-500/30",
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
    <div className="flex items-end justify-between gap-4 mb-6">
      <div className="min-w-0">
        <h2 className="text-xl font-semibold text-white tracking-tight">{title}</h2>
        {subtitle ? <p className="text-sm text-slate-400 mt-1">{subtitle}</p> : null}
      </div>
      {actions ? <div className="flex gap-2 shrink-0">{actions}</div> : null}
    </div>
  );
}

export function StatCard({
  label,
  value,
  icon,
  tone = "slate",
  sub,
  valueClass = "text-white",
}: {
  label: string;
  value: ReactNode;
  icon?: IconName;
  tone?: Tone;
  sub?: ReactNode;
  valueClass?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/70 px-3.5 py-3 min-w-0 transition-colors hover:border-slate-700">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wide text-slate-500 truncate">
          {label}
        </span>
        {icon ? (
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${TONES[tone]}`}>
            <Icon name={icon} className="w-4 h-4" />
          </div>
        ) : null}
      </div>
      <div className={`tnum font-mono text-lg font-semibold mt-1 truncate ${valueClass}`}>{value}</div>
      {sub ? <div className="text-[11px] text-slate-500 mt-0.5 truncate">{sub}</div> : null}
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
      className={`inline-flex items-center gap-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full border ${TONES[tone]} ${BORDERS[tone]}`}
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
  "w-full px-3 py-2 rounded-lg bg-slate-950/70 border border-slate-700/60 text-white text-sm placeholder-slate-500 focus:outline-none focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 transition";

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1.5">
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
