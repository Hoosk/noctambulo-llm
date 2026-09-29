import { useCallback, useEffect, useRef, useState } from "react";
import type { TrainingEvent } from "../api/types";
import { api } from "../api/client";
import {
  appendLogLines,
  applyTrainingEvent,
  EMPTY_JOB,
  mergeStatus,
  type AuthState,
  type Connection,
  type JobInfo,
} from "./jobModel";
import { useLogCursor } from "./useLogCursor";
import { useTrainingStream } from "./useTrainingStream";

/**
 * Owns the training job state: live SSE events, the polling fallback and the
 * worker-log backfill all mutate this hook's own state, so nothing is pushed
 * into a parent through an effect.
 *
 * `jobRef` is the synchronous source of truth: async callbacks (SSE, polling,
 * backfill) read and update it through `updateJob`, so a slow response can
 * never reconcile against a stale snapshot and wipe newer data (e.g. logs).
 */
export function useTrainingJob(authState: AuthState): { job: JobInfo; conn: Connection } {
  const [job, setJob] = useState<JobInfo>(EMPTY_JOB);
  const jobRef = useRef(job);

  const updateJob = useCallback((updater: (prev: JobInfo) => JobInfo) => {
    const next = updater(jobRef.current);
    jobRef.current = next;
    setJob(next);
  }, []);

  // Server-side cursor for worker log lines: live lines arrive via the training
  // SSE; this cursor only drives backfill of what was missed while offline.
  const logCursor = useLogCursor();

  const apply = useCallback(
    (ev: TrainingEvent) => {
      if ("LogEvent" in ev) logCursor.increment();
      updateJob((prev) => applyTrainingEvent(prev, ev));
    },
    [logCursor, updateJob],
  );

  const conn = useTrainingStream(authState, apply);

  // Initial worker-log backfill. From then on the training SSE carries the
  // lines live and the status poll only fetches what was missed.
  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    api
      .getLogs(0)
      .then((res) => {
        if (cancelled) return;
        logCursor.set(res.total);
        updateJob((prev) => appendLogLines(prev, res.lines.map((l) => l.line)));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [authState, logCursor, updateJob]);

  // Polling fallback so metrics update even if the SSE stream is dropped.
  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    let lastPolledEvent = "";

    const backfillLogs = async () => {
      try {
        const res = await api.getLogs(logCursor.get());
        if (cancelled) return;
        logCursor.set(res.total);
        if (res.lines.length > 0) {
          updateJob((prev) => appendLogLines(prev, res.lines.map((l) => l.line)));
        }
      } catch {
        /* ignore transient errors */
      }
    };

    const poll = async () => {
      try {
        const status = await api.getStatus();
        if (cancelled) return;
        const cursor = logCursor.get();
        // A server restart resets the log counter; start over from zero.
        if (status.logs_total < cursor) logCursor.set(0);
        if (status.logs_total > cursor) void backfillLogs();

        const { job: merged, event } = mergeStatus(jobRef.current, status);
        if (event) {
          if ("LogEvent" in event) logCursor.increment();
          const json = JSON.stringify(event);
          if (json === lastPolledEvent) {
            updateJob(() => merged);
            return;
          }
          lastPolledEvent = json;
          updateJob(() => applyTrainingEvent(merged, event));
          return;
        }
        updateJob(() => merged);
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
  }, [authState, logCursor, updateJob]);

  return { job, conn };
}
