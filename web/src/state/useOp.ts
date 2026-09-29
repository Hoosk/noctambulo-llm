import { useEffect, useRef } from "react";
import type { OpStatus } from "../api/types";
import { useAppState } from "./appContext";

/** Live status of one operation, or null when the id is unknown/cleared. */
export function useOp(opId: string | null | undefined): OpStatus | null {
  const { ops } = useAppState();
  if (!opId) return null;
  return ops.find((o) => o.id === opId) ?? null;
}

/** Call `onOutcome` exactly once when the operation reaches a terminal state. */
export function useOpOutcome(
  opId: string | null | undefined,
  onOutcome: (op: OpStatus) => void,
) {
  const { ops } = useAppState();
  const handled = useRef<string | null>(null);
  const callback = useRef(onOutcome);

  useEffect(() => {
    callback.current = onOutcome;
  }, [onOutcome]);

  useEffect(() => {
    if (!opId || handled.current === opId) return;
    const op = ops.find((o) => o.id === opId);
    if (!op || op.status === "running") return;
    handled.current = opId;
    callback.current(op);
  }, [ops, opId]);
}
