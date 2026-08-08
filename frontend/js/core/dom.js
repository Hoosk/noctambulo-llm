// Small DOM helpers.

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, val] of Object.entries(attrs)) {
    if (k === "class") node.className = val;
    else if (k === "text") node.textContent = val;
    else if (k.startsWith("on") && typeof val === "function") {
      node.addEventListener(k.slice(2), val);
    } else if (k === "html") node.innerHTML = val;
    else if (k in node && !(k === "value" && tag === "input" && node.type === "text")) {
      node[k] = val;
    } else {
      node.setAttribute(k, val);
    }
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let i = -1;
  do {
    n /= 1024;
    i++;
  } while (n >= 1024 && i < u.length - 1);
  return `${n.toFixed(1)} ${u[i]}`;
}

export function fmtDur(secs) {
  if (secs == null || !isFinite(secs)) return "—";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}
