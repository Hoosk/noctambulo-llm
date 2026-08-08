// Formatting helpers.

export function fmtLoss(v) {
  return v == null ? "—" : Number(v).toFixed(4);
}

export function fmtLR(v) {
  if (v == null) return "—";
  return Number(v).toExponential(2);
}

export function fmtPct(v) {
  return v == null ? "—" : `${Number(v).toFixed(1)}%`;
}

export function fmtDur(secs) {
  if (secs == null || !isFinite(secs) || secs < 0) return "—";
  const s = Math.floor(secs);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

export function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n;
  let u = -1;
  do {
    v /= 1024;
    u++;
  } while (v >= 1024 && u < units.length - 1);
  return `${v.toFixed(1)} ${units[u]}`;
}

export function fmtNum(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString("es-ES");
}
