import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

type Variant = "primary" | "success" | "danger" | "warn" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary:
    "text-[#140d04] bg-gradient-to-b from-amber-300 to-amber-500 hover:from-amber-200 hover:to-amber-400 shadow-[0_0_0_1px_rgba(201,164,103,0.35),0_10px_30px_rgba(242,169,59,0.16)]",
  success:
    "text-[#07100a] bg-gradient-to-b from-emerald-300 to-emerald-500 hover:from-emerald-200 hover:to-emerald-400 shadow-[0_0_0_1px_rgba(139,195,157,0.3),0_10px_26px_rgba(107,164,124,0.14)]",
  danger:
    "text-white bg-gradient-to-b from-rose-500 to-rose-600 hover:from-rose-400 hover:to-rose-500 shadow-[0_0_0_1px_rgba(207,99,84,0.32),0_10px_26px_rgba(184,72,59,0.18)]",
  warn:
    "text-[#140d04] bg-gradient-to-b from-amber-300 to-amber-500 hover:from-amber-200 hover:to-amber-400 shadow-[0_0_0_1px_rgba(226,154,50,0.32)]",
  ghost:
    "text-slate-300 bg-slate-900/60 hover:bg-slate-800/85 border border-slate-700/70 hover:border-brass-light/45",
};

const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 font-mono text-[11px] font-semibold uppercase tracking-[0.08em] transition duration-200 active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  icon?: IconName;
  children?: ReactNode;
}

export function Button({ variant = "ghost", icon, children, className = "", ...rest }: Props) {
  return (
    <button type="button" className={`${BTN_BASE} ${VARIANTS[variant]} ${className}`} {...rest}>
      {icon ? <Icon name={icon} className="w-4 h-4 shrink-0" /> : null}
      {children}
    </button>
  );
}
