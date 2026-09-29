import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AuthState } from "./jobModel";

/** Periodic authenticated ping: confirms the API is reachable regardless of SSE. */
export function useHeartbeat(authState: AuthState): boolean | null {
  const [pingOk, setPingOk] = useState<boolean | null>(null);

  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    const ping = async () => {
      try {
        await api.ping();
        if (!cancelled) setPingOk(true);
      } catch {
        if (!cancelled) setPingOk(false);
      }
    };
    void ping();
    const timer = window.setInterval(() => void ping(), 10000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [authState]);

  return pingOk;
}
