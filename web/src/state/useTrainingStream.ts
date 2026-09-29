import { useEffect, useState } from "react";
import type { TrainingEvent } from "../api/types";
import type { AuthState, Connection } from "./jobModel";

/**
 * Training event stream (SSE, primary channel) with exponential-backoff
 * reconnect. Returns the connection state for the badge.
 */
export function useTrainingStream(
  authState: AuthState,
  apply: (ev: TrainingEvent) => void,
): Connection {
  const [conn, setConn] = useState<Connection>("connecting");

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

  return conn;
}
