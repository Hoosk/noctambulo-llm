// Kuro-Neko web UI controller.
(function () {
  "use strict";

  // Token: prompt once, then keep in localStorage.
  let TOKEN = localStorage.getItem("kuro_token") || "";
  if (!TOKEN) {
    TOKEN = prompt("Introduce el API token del servidor (X-API-Token):") || "";
    localStorage.setItem("kuro_token", TOKEN);
  }

  const state = {
    chart: null,
    lossHistory: [],   // {step, loss}
    sse: null,
    running: false,
  };

  function headers(json) {
    const h = { "X-API-Token": TOKEN };
    if (json) h["Content-Type"] = "application/json";
    return h;
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (r.status === 401) {
      localStorage.removeItem("kuro_token");
      location.reload();
    }
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t || r.statusText);
    }
    return r.json();
  }

  // ---------- Connectivity ----------
  function setOnline(on) {
    document.getElementById("conn-status").classList.toggle("online", on);
    document.getElementById("conn-status").classList.toggle("offline", !on);
  }

  async function ping() {
    try {
      await api("/api/devices");
      setOnline(true);
    } catch (e) {
      setOnline(false);
    }
  }

  // ---------- Tabs ----------
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      document.getElementById("tab-" + b.dataset.tab).classList.add("active");
    });
  });

  // ---------- Dataset & device dropdowns ----------
  async function loadOptions() {
    try {
      const ds = await api("/api/datasets");
      const sel = document.getElementById("dataset");
      sel.innerHTML = ds
        .map((d) => `<option value="${d}">${d}</option>`)
        .join("");
    } catch (e) {
      document.getElementById("train-msg").textContent = "Error listando datasets: " + e.message;
      document.getElementById("train-msg").className = "msg err";
    }
    try {
      const devs = await api("/api/devices");
      const sel = document.getElementById("device");
      sel.innerHTML = devs
        .map((d) => `<option value="${d.name}">${d.name}</option>`)
        .join("");
      renderDevices(devs);
    } catch (e) {
      document.getElementById("train-msg").textContent = "Error listando devices: " + e.message;
      document.getElementById("train-msg").className = "msg err";
    }
  }

  function renderDevices(devs) {
    const ul = document.getElementById("devices-list");
    ul.innerHTML = devs
      .map((d) => `<li>🖥 ${d.name}</li>`)
      .join("");
  }

  // ---------- Chart ----------
  function initChart() {
    const ctx = document.getElementById("loss-chart");
    state.chart = new Chart(ctx, {
      type: "line",
      data: { datasets: [{ label: "Loss", data: [], borderColor: "#7c6cff", tension: 0.3 }] },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#e6e6ee" } } },
        scales: {
          x: { title: { display: true, text: "step", color: "#8b8fa3" }, ticks: { color: "#8b8fa3" } },
          y: { title: { display: true, text: "loss", color: "#8b8fa3" }, ticks: { color: "#8b8fa3" } },
        },
      },
    });
  }

  function addLossPoint(step, loss) {
    state.lossHistory.push({ step, loss });
    if (state.lossHistory.length > 500) state.lossHistory.shift();
    state.chart.data.datasets[0].data = state.lossHistory;
    state.chart.update();
  }

  // ---------- Metrics render ----------
  function setMetric(id, val) {
    document.getElementById(id).textContent = String(val);
  }

  function fmtDur(secs) {
    if (!secs || secs < 0) return "—";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}m ${s}s`;
  }

  function handleEvent(ev) {
    const k = Object.keys(ev)[0];
    const v = ev[k];
    switch (k) {
      case "Started":
        setMetric("m-state", "Running");
        setMetric("m-epoch", "1/" + v.epochs);
        break;
      case "Progress":
        setMetric("m-state", "Running");
        setMetric("m-epoch", `${v.epoch}/${v.epochs} · ${v.batch}/${v.num_batches}`);
        setMetric("m-loss", v.loss.toFixed(4));
        setMetric("m-avg", v.avg_loss.toFixed(4));
        setMetric("m-lr", v.lr.toExponential(2));
        setMetric("m-tps", v.batches_per_sec.toFixed(1));
        setMetric("m-eta", fmtDur(v.eta_secs));
        addLossPoint(v.step, v.loss);
        break;
      case "Val":
        setMetric("m-ppl", v.ppl.toFixed(2));
        break;
      case "Preview":
        setMetric("m-preview", v.text);
        break;
      case "Done":
        setMetric("m-state", "Completado");
        state.running = false;
        updateButtons();
        break;
      case "Stopped":
        setMetric("m-state", "Parado");
        state.running = false;
        updateButtons();
        break;
      case "Checkpoint":
        setMetric("m-state", `Checkpoint ${v.kind}`);
        break;
    }
  }

  // ---------- SSE ----------
  function connectSSE() {
    if (state.sse) state.sse.close();
    state.sse = new EventSource("/api/train/stream");
    state.sse.addEventListener("training", (msg) => {
      try {
        handleEvent(JSON.parse(msg.data));
      } catch (e) {
        console.error("Bad SSE event", e);
      }
    });
    state.sse.onerror = () => {
      // EventSource auto-reconnects; mark running state from status poll instead.
    };
  }

  // ---------- Train actions ----------
  function updateButtons() {
    document.getElementById("start-btn").disabled = state.running;
    document.getElementById("stop-btn").disabled = !state.running;
  }

  async function startTraining(ev) {
    ev.preventDefault();
    const fields = [
      "file", "device", "epochs", "batchSize", "lr", "maxTokens",
      "nCtx", "nEmbd", "nHead", "nLayer", "valSplit", "gradAccum",
      "saveDir", "outputDir",
    ];
    const body = {};
    for (const f of fields) {
      const el = document.getElementById(f);
      if (f === "maxTokens") body[f] = el.value ? Number(el.value) : null;
      else body[f] = el.value;
    }
    if (body.file === "device") return; // guard
    const msg = document.getElementById("train-msg");
    try {
      await api("/api/train", { method: "POST", headers: headers(true), body: JSON.stringify(body) });
      msg.textContent = "Entrenamiento iniciado.";
      msg.className = "msg ok";
      state.running = true;
      updateButtons();
      state.lossHistory = [];
      connectSSE();
    } catch (e) {
      msg.textContent = "Error: " + e.message;
      msg.className = "msg err";
    }
  }

  async function stopTraining() {
    const msg = document.getElementById("train-msg");
    try {
      await api("/api/train/stop", { method: "POST", headers: headers() });
      msg.textContent = "Solicitando parada...";
      msg.className = "msg ok";
    } catch (e) {
      msg.textContent = "Error: " + e.message;
      msg.className = "msg err";
    }
  }

  // ---------- Init ----------
  document.getElementById("train-form").addEventListener("submit", startTraining);
  document.getElementById("stop-btn").addEventListener("click", stopTraining);

  // Poll status to reflect actual server state (e.g. job running on another tab).
  async function pollStatus() {
    try {
      const s = await api("/api/train/status");
      if (s.running !== state.running) {
        state.running = s.running;
        updateButtons();
        if (s.running) connectSSE();
      }
      if (s.running) setMetric("m-state", "Running");
    } catch (e) {
      /* ignore */
    }
    setTimeout(pollStatus, 3000);
  }

  initChart();
  loadOptions();
  connectSSE();
  pollStatus();
  ping();
})();
