// API client: single place for fetch logic + token handling.

const TOKEN_KEY = "kuro_neko_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

function buildHeaders(json) {
  const h = new Headers();
  const token = getToken();
  if (token) h.set("X-API-Token", token);
  if (json) h.set("Content-Type", "application/json");
  return h;
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: buildHeaders(options.json),
    ...options,
  });

  if (res.status === 401) {
    // Token revoked/expired mid-session: drop it and let the app show the gate.
    setToken("");
    window.dispatchEvent(new CustomEvent("kuro:unauthorized"));
    throw new ApiError(401, "Token inválido o ausente");
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.error) detail = body.error;
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  get(path) {
    return request(path);
  },
  ping() {
    return request("/api/ping");
  },
  getLogs(since = 0) {
    return request(`/api/train/logs?since=${since}`);
  },
  getCheckpoints(dir = "checkpoints_web") {
    return request(`/api/checkpoints?dir=${encodeURIComponent(dir)}`);
  },
  post(path, body) {
    return request(path, { method: "POST", json: true, body: JSON.stringify(body) });
  },
  delete(path) {
    return request(path, { method: "DELETE" });
  },
};
