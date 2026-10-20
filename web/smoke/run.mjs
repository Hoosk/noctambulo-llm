#!/usr/bin/env node
// CDP browser smoke test for the noctambulo web console.
//
// Prerequisites:
//   1. Server running:      cargo run --release -- serve --token <token> --datasets-dir datasets --models-dir .
//   2. Headless Chromium:   chromium --headless=new --remote-debugging-port=9222 --user-data-dir=/tmp/noctambulo-smoke-profile about:blank
//   3. Node 22+ (global fetch + WebSocket).
//
// Usage:
//   node web/smoke/run.mjs [--url http://127.0.0.1:8080/] [--token noctambulo-demo]
//                          [--cdp http://127.0.0.1:9222] [--models-dir .] [--datasets-dir datasets]
//                          [--fixture MODEL_DIR] [--skip-training] [--keep-artifacts]
//
// The suite creates a disposable copy of the fixture model, a small dataset mix and a tiny
// training run; everything it creates is removed afterwards unless --keep-artifacts is set.

import { cpSync, existsSync, readdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "..", "..");

function parseArgs(argv) {
  const opts = {
    url: "http://127.0.0.1:8080/",
    token: "noctambulo-demo",
    cdp: "http://127.0.0.1:9222",
    modelsDir: REPO,
    datasetsDir: resolve(REPO, "datasets"),
    fixture: "",
    skipTraining: false,
    keepArtifacts: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--url") opts.url = value;
    else if (key === "--token") opts.token = value;
    else if (key === "--cdp") opts.cdp = value.replace(/\/$/, "");
    else if (key === "--models-dir") opts.modelsDir = resolve(value);
    else if (key === "--datasets-dir") opts.datasetsDir = resolve(value);
    else if (key === "--fixture") opts.fixture = value;
    else if (key === "--skip-training") opts.skipTraining = true;
    else if (key === "--keep-artifacts") opts.keepArtifacts = true;
    else if (key === "--help") {
      console.log(
        "usage: node web/smoke/run.mjs [--url URL] [--token TOKEN] [--cdp URL] " +
          "[--models-dir DIR] [--datasets-dir DIR] [--fixture NAME] [--skip-training] [--keep-artifacts]",
      );
      process.exit(0);
    } else if (key.startsWith("--")) fail(`unknown flag ${key}`);
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const API = opts.url.replace(/\/$/, "");
const SCRATCH = "smoke_scratch_model";
const SCRATCH_RENAMED = "smoke_renamed_model";
const MIX_FILE = "smoke_mix_ui.txt";
const RUN_DIR = "smoke_ui_run";

const results = [];
let failed = 0;
function fail(message) {
  console.error(`fatal: ${message}`);
  process.exit(2);
}
function record(name, ok, detail = "") {
  results.push({ name, ok, detail });
  if (!ok) failed += 1;
  console.log(`${ok ? "\u2714" : "\u2718"} ${name}${detail ? ` — ${detail}` : ""}`);
}

// ---------------------------------------------------------------- CDP client
const targetsRes = await fetch(`${opts.cdp}/json/list`);
if (!targetsRes.ok) {
  throw new Error(`CDP target list failed: HTTP ${targetsRes.status} ${targetsRes.statusText}`);
}
const targets = await targetsRes.json();
const page = targets.find((t) => t.type === "page");
if (!page) fail(`no page target on ${opts.cdp}`);
const ws = new WebSocket(page.webSocketDebuggerUrl);
let msgId = 0;
const pending = new Map();
const consoleErrors = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  } else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") {
    const text = (m.params.args ?? []).map((a) => a.value ?? a.description ?? "").join(" ");
    if (!/favicon/i.test(text)) consoleErrors.push(text);
  }
});
await new Promise((res, rej) => {
  ws.addEventListener("open", res);
  ws.addEventListener("error", rej);
});
const send = (method, params = {}) =>
  new Promise((res) => {
    const id = ++msgId;
    pending.set(id, res);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => {
  const reply = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (reply.result?.exceptionDetails) {
    throw new Error(reply.result.exceptionDetails.exception?.description ?? "evaluate failed");
  }
  return reply.result?.result?.value;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(expression, timeoutMs = 60_000, label = expression) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await evaluate(expression)) return true;
    if (Date.now() > deadline) throw new Error(`timeout waiting for ${label}`);
    await sleep(400);
  }
}
const clickText = (scope, text) =>
  evaluate(
    `[...document.querySelectorAll(${JSON.stringify(scope)})].find((b) => b.textContent.trim() === ${JSON.stringify(text)})?.click(), true`,
  );
const bodyText = () => evaluate(`document.body.innerText`);

await send("Runtime.enable");
await send("Page.enable");

// ------------------------------------------------------------------- helpers
const apiFetch = (path, init = {}) =>
  fetch(`${API}${path}`, {
    ...init,
    headers: { "X-API-Token": opts.token, "Content-Type": "application/json", ...(init.headers ?? {}) },
  });

async function clickNav(name) {
  await evaluate(
    `[...document.querySelectorAll('nav button')].find((b) => b.textContent.trim() === ${JSON.stringify(name)})?.click()`,
  );
  await sleep(900);
}

async function setField(label, value) {
  const ok = await evaluate(`(() => {
    const field = [...document.querySelectorAll('label')].find(
      (l) =>
        (l.querySelector(':scope > span')?.textContent ?? l.textContent).trim() === ${JSON.stringify(label)},
    );
    const el =
      field?.querySelector('input, select, textarea') ??
      (field?.htmlFor ? document.getElementById(field.htmlFor) : null);
    if (!el) return false;
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype
      : el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, ${JSON.stringify(value)});
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);
  if (!ok) throw new Error(`field not found: ${label}`);
  await sleep(200);
}

async function toggleDataset(name) {
  const ok = await evaluate(`(() => {
    const row = [...document.querySelectorAll('main div')].find(
      (d) => d.querySelector(':scope > input[type=checkbox]') && d.textContent.includes(${JSON.stringify(name)}),
    );
    if (!row) return false;
    row.querySelector(':scope > input[type=checkbox]').click();
    return true;
  })()`);
  if (!ok) throw new Error(`dataset row not found: ${name}`);
  await sleep(300);
}

// ------------------------------------------------------------------- cleanup
function cleanup() {
  if (opts.keepArtifacts) {
    console.log("artifacts kept (--keep-artifacts)");
    return;
  }
  const paths = [
    resolve(opts.modelsDir, SCRATCH),
    resolve(opts.modelsDir, SCRATCH_RENAMED),
    resolve(opts.modelsDir, `${SCRATCH}_imported`),
    resolve(opts.modelsDir, RUN_DIR),
    resolve(opts.datasetsDir, MIX_FILE),
  ];
  for (const p of paths) rmSync(p, { recursive: true, force: true });
  if (existsSync(opts.modelsDir)) {
    for (const f of readdirSync(opts.modelsDir)) {
      if ((f.startsWith(`${SCRATCH}_`) || f.startsWith(`${SCRATCH_RENAMED}`)) && f.endsWith(".gguf")) {
        rmSync(resolve(opts.modelsDir, f), { force: true });
      }
    }
  }
}

// ------------------------------------------------------------------- the run
try {
  await evaluate(`location.href = ${JSON.stringify(opts.url)}`);
  await sleep(2500);
  if (await evaluate(`Boolean(document.querySelector('#token-input'))`)) {
    await evaluate(`(() => {
      const i = document.querySelector('#token-input');
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(i, ${JSON.stringify(opts.token)});
      i.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await clickText("button", "Connect");
    await sleep(1500);
  }
  record("login", await evaluate(`document.querySelectorAll('nav button').length >= 5`));

  // --- D4: Generate view exposes RoPE scaling controls ---------------------
  await clickNav("Generate");
  await setField("RoPE scaling", "linear");
  await setField("RoPE factor", "2");
  const ropeState = await evaluate(`(() => {
    const field = [...document.querySelectorAll('main label')].find(
      (l) => l.querySelector(':scope > span')?.textContent.trim() === 'RoPE scaling',
    );
    const select = field?.querySelector('select');
    const factorField = [...document.querySelectorAll('main label')].find(
      (l) => l.querySelector(':scope > span')?.textContent.trim() === 'RoPE factor',
    );
    return { kind: select?.value, factor: factorField?.querySelector('input')?.value };
  })()`);
  record(
    "D4 generate RoPE fields",
    ropeState.kind === "linear" && ropeState.factor === "2",
    `kind=${ropeState.kind} factor=${ropeState.factor}`,
  );

  // --- D6: Train form exposes optimizer/loss controls ----------------------
  await clickNav("Train");
  // The wizard mounts only the active step; open the Training step first.
  await evaluate(
    `[...document.querySelectorAll('main button')].find((b) => b.textContent.includes('Train'))?.click()`,
  );
  await sleep(400);
  await setField("Weight decay", "0.2");
  await setField("Label smoothing", "0.1");
  await setField("Z-loss", "0.001");
  const lossState = await evaluate(`(() => {
    const read = (label) => {
      const field = [...document.querySelectorAll('main label')].find(
        (l) => l.querySelector(':scope > span')?.textContent.trim() === label,
      );
      return field?.querySelector('input')?.value;
    };
    return { wd: read('Weight decay'), ls: read('Label smoothing'), zl: read('Z-loss') };
  })()`);
  record(
    "D6 train loss fields",
    lossState.wd === "0.2" && lossState.ls === "0.1" && lossState.zl === "0.001",
    JSON.stringify(lossState),
  );

  // --- E4: Devices telemetry card ------------------------------------------
  await clickNav("Devices");
  const devices = await bodyText();
  record(
    "E4 telemetry card",
    /live telemetry/i.test(devices) && /system memory/i.test(devices) && /process cpu/i.test(devices),
    "system + process counters",
  );
  const systemOk = /system/i.test(devices) && /cores/i.test(devices) && /kernel/i.test(devices);
  record("E4b host system card", systemOk, "CPU, cores, OS/kernel");

  // --- E6/E3: weighted combine, .bin rejection, share %, op progress -------
  await clickNav("Datasets");
  await toggleDataset("combined_corpus.txt");
  await toggleDataset("weighted_mix_web.txt");
  await waitFor(`document.body.innerText.includes('Equalize weights')`, 15_000, "combine card");
  const shares = await evaluate(`document.body.innerText.match(/\\d+%/g)?.length ?? 0`);
  record("E6 combine share display", shares >= 2, `${shares} share values`);

  await toggleDataset("500k_copy.bin");
  const blocked = await waitFor(
    `document.body.innerText.includes('Binary datasets cannot be mixed')`,
    15_000,
    "bin rejection notice",
  )
    .then(() => true)
    .catch(() => false);
  record("E6 combine blocks .bin", blocked, "explicit error shown");
  await toggleDataset("500k_copy.bin");

  await setField("Mix name", MIX_FILE);
  await setField("Max size (MB · 0 = all)", "1");
  await clickText("main button", "Combine");
  const mixOk = await waitFor(`document.body.innerText.includes('Mix created')`, 180_000, "mix result")
    .then(() => true)
    .catch(() => false);
  record("E3 combine op progress + result", mixOk && existsSync(resolve(opts.datasetsDir, MIX_FILE)), MIX_FILE);

  // --- E5/E7: training run, run history, worker logs, val curve ------------
  if (!opts.skipTraining) {
    const trainBody = {
      file: MIX_FILE,
      epochs: 1,
      batchSize: 4,
      lr: 0.0003,
      maxTokens: 12000,
      valSplit: 0.1,
      nCtx: 128,
      nEmbd: 128,
      nHead: 4,
      nKvHead: 2,
      nLayer: 2,
      outputDir: RUN_DIR,
      device: "DefaultDevice",
      dtype: "f32",
      weightDecay: 0.1,
      labelSmoothing: 0.0,
      zLoss: 0.0,
    };
    const start = await apiFetch("/api/train", { method: "POST", body: JSON.stringify(trainBody) });
    record("E5 start training job", start.ok, `HTTP ${start.status}`);

    const deadline = Date.now() + 15 * 60_000;
    let status = null;
    for (;;) {
      status = await (await apiFetch("/api/train/status")).json();
      if (!status.running) break;
      if (Date.now() > deadline) throw new Error("timeout waiting for training");
      await sleep(2000);
    }
    record("E7 training finished with val points", status.val_points.length >= 1, `${status.val_points.length} val point(s)`);

    await clickNav("Train");
    await clickText("main button", "Refresh");
    await waitFor(`document.body.innerText.toLowerCase().includes('completed')`, 60_000, "run status badge");
    const runInfo = await evaluate(`(() => {
      const checks = [...document.querySelectorAll('main input[type=checkbox]')];
      const runCheck = checks.find((c) => c.closest('li')?.textContent.includes(${JSON.stringify(MIX_FILE)}));
      runCheck?.click();
      return Boolean(runCheck);
    })()`);
    await sleep(1200);
    const canvases = await evaluate(`document.querySelectorAll('main canvas').length`);
    record("E5 run history + overlay curve", runInfo && canvases >= 1, `${canvases} canvases`);

    const logs = await evaluate(`(() => {
      const pre = [...document.querySelectorAll('main pre')].find((p) => p.querySelector('span'));
      return pre ? pre.querySelectorAll('span').length : 0;
    })()`);
    record("E7 worker logs", logs >= 1, `${logs} log lines`);
  }

  // --- E2/E3: model files, rename, export, delete --------------------------
  // The fixture is the tiny model just trained by this suite (new-architecture
  // checkpoint); pass --fixture to use another model directory instead.
  const fixturePath = opts.fixture
    ? resolve(opts.modelsDir, opts.fixture)
    : resolve(opts.modelsDir, RUN_DIR);
  const crud = existsSync(fixturePath);
  if (crud) {
    rmSync(resolve(opts.modelsDir, SCRATCH), { recursive: true, force: true });
    cpSync(fixturePath, resolve(opts.modelsDir, SCRATCH), { recursive: true });
  }

  await clickNav("Models");
  if (crud) {
    await waitFor(`document.body.innerText.includes(${JSON.stringify(SCRATCH)})`, 30_000, "scratch model row");
    await evaluate(`(() => {
      const row = [...document.querySelectorAll('main li')].find((li) => li.textContent.includes(${JSON.stringify(SCRATCH)}));
      [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Files').click();
    })()`);
    await waitFor(`document.body.innerText.includes('model_weights.bin')`, 30_000, "files panel");
    const filesOk = await evaluate(`(() => {
      const row = [...document.querySelectorAll('main li')].find((li) => li.textContent.includes(${JSON.stringify(SCRATCH)}));
      return row.querySelectorAll('a[download]').length;
    })()`);
    record("E2 model files + download", filesOk > 0, `${filesOk} download links`);

    await evaluate(`(() => {
      const row = [...document.querySelectorAll('main li')].find((li) => li.textContent.includes(${JSON.stringify(SCRATCH)}));
      [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Rename').click();
    })()`);
    await waitFor(`Boolean(document.querySelector('#prompt-input'))`, 15_000, "rename dialog");
    await setField("New name", SCRATCH_RENAMED);
    await clickText("button", "Save");
    await waitFor(`document.body.innerText.includes(${JSON.stringify(SCRATCH_RENAMED)})`, 30_000, "renamed row");
    record("E2 rename model", true, `renamed to ${SCRATCH_RENAMED}`);

    // E3: export op with progress, run before deleting the scratch model.
    await evaluate(`(() => {
      const row = [...document.querySelectorAll('main li')].find((li) => li.textContent.includes(${JSON.stringify(SCRATCH_RENAMED)}));
      [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Export GGUF').click();
    })()`);
    const exportOk = await waitFor(
      `document.body.innerText.includes('.gguf')`,
      180_000,
      "export result",
    )
      .then(() => true)
      .catch(() => false);
    record("E3 export op progress + result", exportOk, "gguf produced");

    // E8: import the GGUF just exported and verify the model list marks it compatible.
    const importedName = `${SCRATCH}_imported`;
    const impRes = await apiFetch("/api/models/import", {
      method: "POST",
      body: JSON.stringify({
        ggufPath: `exported/${SCRATCH_RENAMED}.gguf`,
        outputDir: importedName,
      }),
    });
    const impBody = await impRes.json().catch(() => null);
    const modelsRes = await apiFetch("/api/models");
    const modelsList = await modelsRes.json().catch(() => []);
    const imported = modelsList.find((m) => m.name === importedName);
    record(
      "E8 import GGUF",
      impRes.ok && imported?.compatible === true && imported?.arch_version === 4,
      impBody?.error ?? `compatible=${imported?.compatible}`,
    );
    await apiFetch(`/api/models/${encodeURIComponent(importedName)}`, { method: "DELETE" });

    await evaluate(`(() => {
      const row = [...document.querySelectorAll('main li')].find((li) => li.textContent.includes(${JSON.stringify(SCRATCH_RENAMED)}));
      row.querySelector('button[aria-label^="Delete"]').click();
    })()`);
    await waitFor(`document.body.innerText.includes('Delete model?')`, 15_000, "delete dialog");
    await evaluate(`(() => {
      const overlay = [...document.querySelectorAll('div.fixed')].find((d) => d.textContent.includes('Delete model?'));
      [...overlay.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Delete').click();
    })()`);
    await waitFor(
      `![...document.querySelectorAll('main li')].some((li) => li.textContent.includes(${JSON.stringify(SCRATCH_RENAMED)}))`,
      30_000,
      "row removed",
    );
    record("E2 delete model", true, "scratch model removed");
  } else {
    record("E2 model CRUD", false, `fixture not found: ${fixturePath}`);
  }

  // --- HTTP contracts ------------------------------------------------------
  const index = await fetch(opts.url);
  record("serve index 200", index.status === 200, `HTTP ${index.status}`);
  const spa = await fetch(`${API}/models`);
  record("SPA fallback 200", spa.status === 200 && (await spa.text()).includes("<div id=\"root\""), `HTTP ${spa.status}`);
  const unknown = await apiFetch("/api/definitely-not-a-route");
  const unknownJson = await unknown.json().catch(() => null);
  record("unknown API 404 JSON", unknown.status === 404 && unknownJson?.error != null, `HTTP ${unknown.status}`);

  record("no unexpected console errors", consoleErrors.length === 0, consoleErrors.join(" | ").slice(0, 120));
} catch (error) {
  record("suite aborted", false, error.message);
} finally {
  // Stop a leftover job and remove artifacts.
  try {
    const status = await (await apiFetch("/api/train/status")).json();
    if (status.running) await apiFetch("/api/train/stop", { method: "POST" });
  } catch {
    /* server may be down; ignore */
  }
  cleanup();
  ws.close();
}

const total = results.length;
console.log(`\n${total - failed}/${total} steps ok`);
process.exit(failed === 0 ? 0 : 1);
