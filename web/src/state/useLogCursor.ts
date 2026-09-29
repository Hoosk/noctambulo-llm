import { useCallback, useMemo, useRef } from "react";

export interface LogCursor {
  get: () => number;
  set: (value: number) => void;
  increment: (by?: number) => void;
}

/**
 * Mutable cursor for the worker-log backfill, wrapped so the ref never escapes
 * the hook that owns it (the React compiler forbids mutating refs passed as
 * hook arguments).
 */
export function useLogCursor(): LogCursor {
  const cursor = useRef(0);
  const get = useCallback(() => cursor.current, []);
  const set = useCallback((value: number) => {
    cursor.current = value;
  }, []);
  const increment = useCallback((by = 1) => {
    cursor.current += by;
  }, []);
  return useMemo(() => ({ get, set, increment }), [get, set, increment]);
}
