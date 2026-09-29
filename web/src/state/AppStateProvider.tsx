import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { ApiError, api, UNAUTHORIZED_EVENT } from "../api/client";
import type { TrainingEvent } from "../api/types";
import { AppStateContext, type AppStateValue } from "./appContext";
import {
  applyTrainingEvent,
  EMPTY_JOB,
  mergeStatus,
  type AuthState,
  type Connection,
  type JobInfo,
} from "./jobModel";

const ACTIVE_MODEL_KEY = "kuro_active_model";

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [conn, setConn] = useState<Connection>("connecting");
  const [job, setJob] = useState<JobInfo>(EMPTY_JOB);
  const [pingOk, setPingOk] = useState<boolean | null>(null);
  const [activeModel, setActiveModelState] = useState(
    () => localStorage.getItem(ACTIVE_MODEL_KEY) ?? "",
  );

  const setActiveModel = useCallback((dir: string) => {
    if (dir) localStorage.setItem(ACTIVE_MODEL_KEY, dir);
    else localStorage.removeItem(ACTIVE_MODEL_KEY);
    setActiveModelState(dir);
  }, []);

  const jobRef = useRef(job);
  useEffect(() => {
    jobRef.current = job;
  }, [job]);

  const apply = useCallback((ev: TrainingEvent) => {
    setJob((prev) => applyTrainingEvent(prev, ev));
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

  // SSE stream (primary channel) with exponential-backoff reconnect.
  useEffect(() => {
    if (authState !== "authed") return;
    const ctrl = new AbortController();
    const decoder = new TextDecoder();
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let buffer = "";
    let retry = 2000;
    let timer: number | null = null;

    const feed = (text: string) => {
      buffer += text;
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        const json = dataLine.slice(5).trim();
        if (!json) continue;
        try {
          apply(JSON.parse(json) as TrainingEvent);
        } catch (err) {
          console.error("Bad SSE payload", json, err);
        }
      }
    };

    const schedule = () => {
      if (ctrl.signal.aborted || timer != null) return;
      timer = window.setTimeout(() => {
        timer = null;
        connect();
      }, retry);
      retry = Math.min(retry * 2, 15000);
    };

    const pump = () => {
      reader!
        .read()
        .then(({ done, value }) => {
          if (ctrl.signal.aborted) return;
          if (done) {
            setConn("offline");
            schedule();
            return;
          }
          feed(decoder.decode(value, { stream: true }));
          pump();
        })
        .catch((err: unknown) => {
          if (ctrl.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
          setConn("offline");
          schedule();
        });
    };

    const connect = () => {
      setConn("connecting");
      fetch("/api/train/stream", { signal: ctrl.signal })
        .then((res) => {
          if (ctrl.signal.aborted) return;
          if (!res.ok || !res.body) {
            setConn("offline");
            schedule();
            return;
          }
          setConn("online");
          retry = 2000;
          reader = res.body.getReader();
          pump();
        })
        .catch((err: unknown) => {
          if (ctrl.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) return;
          setConn("offline");
          schedule();
        });
    };

    connect();
    return () => {
      ctrl.abort();
      if (timer != null) window.clearTimeout(timer);
      reader?.cancel().catch(() => {});
      setConn("connecting");
    };
  }, [authState, apply]);

  // Polling fallback so metrics update even if the SSE stream is dropped.
  const lastPolledEvent = useRef("");
  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await api.getStatus();
        if (cancelled) return;
        const { job: merged, event } = mergeStatus(jobRef.current, status);
        if (event) {
          const json = JSON.stringify(event);
          if (json === lastPolledEvent.current) {
            setJob(merged);
            return;
          }
          lastPolledEvent.current = json;
          setJob(applyTrainingEvent(merged, event));
          return;
        }
        setJob(merged);
      } catch {
        /* ignore transient errors */
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [authState]);

  // Heartbeat: confirm the API is reachable regardless of SSE state.
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

  const value = useMemo<AppStateValue>(
    () => ({
      authState,
      submitToken,
      logout,
      conn: conn === "online" || pingOk ? "online" : pingOk === false ? "offline" : "connecting",
      job,
      activeModel,
      setActiveModel,
    }),
    [authState, submitToken, logout, conn, pingOk, job, activeModel, setActiveModel],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}
