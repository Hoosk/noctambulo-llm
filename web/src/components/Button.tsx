import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon, type IconName } from "./Icon";

type Variant = "primary" | "success" | "danger" | "warn" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary:
    "text-white bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 shadow-lg shadow-violet-600/25",
  success: "text-white bg-emerald-600 hover:bg-emerald-500 shadow-lg shadow-emerald-600/20",
  danger:
    "text-white bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 shadow-lg shadow-rose-600/20",
  warn: "text-white bg-amber-600 hover:bg-amber-500 shadow-lg shadow-amber-600/20",
  ghost: "text-slate-300 bg-slate-800/70 hover:bg-slate-700/70 border border-slate-700/60",
};

const BTN_BASE =
  "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none";

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
