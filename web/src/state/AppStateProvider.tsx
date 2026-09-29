import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { ApiError, api, UNAUTHORIZED_EVENT } from "../api/client";
import { AppStateContext, type AppStateValue } from "./appContext";
import type { AuthState } from "./jobModel";
import { useHeartbeat } from "./useHeartbeat";
import { useOpsStream } from "./useOpsStream";
import { useTrainingJob } from "./useTrainingJob";

const ACTIVE_MODEL_KEY = "kuro_active_model";

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [activeModel, setActiveModelState] = useState(
    () => localStorage.getItem(ACTIVE_MODEL_KEY) ?? "",
  );

  const setActiveModel = useCallback((dir: string) => {
    if (dir) localStorage.setItem(ACTIVE_MODEL_KEY, dir);
    else localStorage.removeItem(ACTIVE_MODEL_KEY);
    setActiveModelState(dir);
  }, []);

  const submitToken = useCallback(async (candidate: string) => {
    const value = candidate.trim();
    if (!value) return false;
    try {
      await api.login(value);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) return false;
      throw err;
    }
    setAuthState("authed");
    return true;
  }, []);

  const logout = useCallback(() => {
    void api.logout().catch(() => {});
    setAuthState("gate");
  }, []);

  // Initial session check: the HttpOnly cookie set by a previous login (if any)
  // travels with this request.
  useEffect(() => {
    if (authState !== "checking") return;
    let cancelled = false;
    api
      .ping()
      .then(() => {
        if (!cancelled) setAuthState("authed");
      })
      .catch(() => {
        if (!cancelled) setAuthState("gate");
      });
    return () => {
      cancelled = true;
    };
  }, [authState]);

  // A 401 anywhere (session revoked/expired) drops back to the gate.
  useEffect(() => {
    const onUnauthorized = () => setAuthState("gate");
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, []);

  const { job, conn } = useTrainingJob(authState);
  const { ops, cancelOp } = useOpsStream(authState);
  const pingOk = useHeartbeat(authState);

  const value = useMemo<AppStateValue>(
    () => ({
      authState,
      submitToken,
      logout,
      conn: conn === "online" || pingOk ? "online" : pingOk === false ? "offline" : "connecting",
      job,
      ops,
      cancelOp,
      activeModel,
      setActiveModel,
    }),
    [authState, submitToken, logout, conn, pingOk, job, ops, cancelOp, activeModel, setActiveModel],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}
