import { useCallback, useEffect, useState } from "react";
import type { OpEvent, OpStatus } from "../api/types";
import { api } from "../api/client";
import type { AuthState } from "./jobModel";
import { applyOpEvent } from "./opsModel";

/**
 * Owns the background-operations state and subscribes to the ops SSE stream,
 * with a snapshot backfill on every (re)connect.
 */
export function useOpsStream(authState: AuthState): {
  ops: OpStatus[];
  cancelOp: (id: string) => void;
} {
  const [ops, setOps] = useState<OpStatus[]>([]);

  const applyOp = useCallback((ev: OpEvent) => {
    setOps((prev) => applyOpEvent(prev, ev));
  }, []);

  const cancelOp = useCallback((id: string) => {
    void api.cancelOp(id).catch(() => {});
  }, []);

  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    const ctrl = new AbortController();
    const decoder = new TextDecoder();
    let retry = 2000;
    let timer: number | null = null;

    const refresh = () => {
      api
        .getOps()
        .then((list) => {
          if (!cancelled) setOps(list);
        })
        .catch(() => {});
    };

    const schedule = () => {
      if (cancelled || timer != null) return;
      timer = window.setTimeout(() => {
        timer = null;
        void connect();
      }, retry);
      retry = Math.min(retry * 2, 15000);
    };

    const connect = async () => {
      try {
        const res = await fetch("/api/ops/stream", { signal: ctrl.signal });
        if (cancelled) return;
        if (!res.ok || !res.body) {
          schedule();
          return;
        }
        retry = 2000;
        refresh();
        let buffer = "";
        const reader = res.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (cancelled) return;
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let boundary: number;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const chunk = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
            if (!dataLine) continue;
            const json = dataLine.slice(5).trim();
            if (!json) continue;
            try {
              applyOp(JSON.parse(json) as OpEvent);
            } catch {
              /* ignore malformed payloads */
            }
          }
        }
        schedule();
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) return;
        schedule();
      }
    };

    void connect();
    return () => {
      cancelled = true;
      ctrl.abort();
      if (timer != null) window.clearTimeout(timer);
    };
  }, [authState, applyOp]);

  return { ops, cancelOp };
}
