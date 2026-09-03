// SSE client using fetch + ReadableStream.
// EventSource cannot send custom headers (we need X-API-Token), so we parse
// the SSE stream manually.

import { getToken, ApiError } from "./api.js";
import { emit } from "./events.js";

let controller = null;
let retryTimer = null;
let retryDelay = 2000;

export function isConnected() {
  return controller !== null;
}

function scheduleReconnect() {
  if (retryTimer || controller) return;
  retryTimer = setTimeout(() => {
    retryTimer = null;
    connect();
  }, retryDelay);
  retryDelay = Math.min(retryDelay * 2, 15000);
}

export function connect() {
  disconnect();

  const ctrl = new AbortController();
  controller = ctrl;

  (async () => {
    let buffer = "";
    try {
      const res = await fetch("/api/train/stream", {
        headers: { "X-API-Token": getToken() },
        signal: ctrl.signal,
      });

      if (!res.ok) {
        throw new ApiError(res.status, "SSE stream error");
      }
      if (!res.body) throw new Error("No response body");

      // Only mark as connected once the stream actually opened, and reset the
      // reconnect backoff after a successful (re)connect.
      emit("sse:status", { connected: true });
      retryDelay = 2000;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on SSE event boundary "\n\n" and dispatch "data:" lines.
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const dataLine = chunk
            .split("\n")
            .find((l) => l.startsWith("data:"));
          if (dataLine) {
            const json = dataLine.slice(5).trim();
            if (json) {
              try {
                emit("training:event", JSON.parse(json));
              } catch (e) {
                console.error("Bad SSE payload", json, e);
              }
            }
          }
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        emit("sse:error", e);
        scheduleReconnect();
      }
    } finally {
      if (controller === ctrl) {
        controller = null;
        emit("sse:status", { connected: false });
      }
    }
  })();
}

export function disconnect() {
  if (controller) {
    controller.abort();
    controller = null;
    emit("sse:status", { connected: false });
  }
  if (retryTimer) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  retryDelay = 2000;
}
