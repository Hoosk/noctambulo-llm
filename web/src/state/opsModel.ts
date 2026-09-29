import type { OpEvent, OpStatus } from "../api/types";

/** Finished operations kept in the client, matching the server-side cap. */
export const MAX_OPS = 60;

/** `null` while the total is unknown (server streaming without a length). */
export function opPercent(op: OpStatus): number | null {
  if (op.total <= 0) return null;
  return Math.min(100, (op.done / op.total) * 100);
}

export function applyOpEvent(ops: OpStatus[], ev: OpEvent): OpStatus[] {
  if ("Started" in ev) {
    const v = ev.Started;
    const next = ops.filter((o) => o.id !== v.id);
    next.unshift({
      id: v.id,
      kind: v.kind,
      label: v.label,
      phase: "starting",
      done: 0,
      total: 0,
      unit: "",
      status: "running",
      message: null,
      error: null,
      result: null,
      started_unix: v.started_unix,
      finished_unix: null,
    });
    return next.slice(0, MAX_OPS);
  }

  const id =
    "Progress" in ev
      ? ev.Progress.id
      : "Done" in ev
        ? ev.Done.id
        : "Failed" in ev
          ? ev.Failed.id
          : ev.Cancelled.id;

  const finishedAt = Math.floor(Date.now() / 1000);
  return ops.map((op) => {
    if (op.id !== id) return op;
    if ("Progress" in ev) {
      const v = ev.Progress;
      return { ...op, phase: v.phase, done: v.done, total: v.total, unit: v.unit };
    }
    if ("Done" in ev) {
      const v = ev.Done;
      return {
        ...op,
        status: "done" as const,
        phase: "done",
        message: v.message,
        result: v.result,
        finished_unix: finishedAt,
        done: op.total > 0 ? op.total : op.done,
      };
    }
    if ("Failed" in ev) {
      return {
        ...op,
        status: "failed" as const,
        phase: "failed",
        error: ev.Failed.error,
        finished_unix: finishedAt,
      };
    }
    return {
      ...op,
      status: "cancelled" as const,
      phase: "cancelled",
      message: "cancelled",
      finished_unix: finishedAt,
    };
  });
}
