import { useCallback, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Icon } from "../components/Icon";
import type { Tone } from "../components/ui";
import { ToastContext, type ToastApi } from "./toastContext";

interface ToastItem {
  id: number;
  message: string;
  tone: Tone;
  icon: "check" | "alert" | "info";
}

const TONES: Record<Tone, string> = {
  slate: "bg-slate-700/40 text-slate-300",
  violet: "bg-violet-500/15 text-violet-300",
  fuchsia: "bg-fuchsia-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (message: string, tone: Tone, icon: ToastItem["icon"]) => {
      const id = nextId.current++;
      setToasts((prev) => [...prev, { id, message, tone, icon }]);
      setTimeout(() => dismiss(id), 4200);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      success: (m) => push(m, "emerald", "check"),
      error: (m) => push(m, "rose", "alert"),
      info: (m) => push(m, "violet", "info"),
    }),
    [push],
  );

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div className="fixed bottom-4 right-4 z-[70] flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)]">
        {toasts.map((t) => (
          <div key={t.id} className="glass-card rounded-xl p-3.5 flex items-start gap-3 animate-fade-slide">
            <div
              className={`shrink-0 w-7 h-7 rounded-lg flex items-center justify-center ${TONES[t.tone]}`}
            >
              <Icon name={t.icon} className="w-3.5 h-3.5" />
            </div>
            <p className="text-sm text-slate-200 flex-1 min-w-0 break-words">{t.message}</p>
            <button
              type="button"
              className="text-slate-500 hover:text-white transition shrink-0"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss"
            >
              <Icon name="x" className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
