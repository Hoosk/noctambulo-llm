// Lightweight wrapper around Chart.js for the loss curve.
// Chart.js must be initialized only after the canvas is attached to the DOM,
// otherwise it measures a wrong (huge) container and breaks the layout.

import { el } from "../core/dom.js";

export default class LossChart {
  constructor() {
    this.history = []; // {x: step, y: loss}
    this.chart = null;
    this.canvas = null;
  }

  // Creates the DOM node. Call attach() once it is in the document.
  build() {
    const container = el("div", { class: "relative h-64 w-full overflow-hidden" });
    container.innerHTML = `<canvas></canvas>`;
    this.canvas = container.querySelector("canvas");
    return container;
  }

  // Initialize Chart.js now that the canvas is mounted. Idempotent.
  attach() {
    if (this.chart || !this.canvas) return;
    const ctx = this.canvas.getContext("2d");

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Loss",
            data: this.history,
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139,92,246,0.08)",
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0f172a",
            borderColor: "#334155",
            borderWidth: 1,
            titleColor: "#e2e8f0",
            bodyColor: "#94a3b8",
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#64748b", maxTicksLimit: 10 },
            title: { display: true, text: "step", color: "#64748b" },
          },
          y: {
            grid: { color: "rgba(51,65,85,0.3)" },
            ticks: { color: "#64748b" },
            title: { display: true, text: "loss", color: "#64748b" },
          },
        },
      },
    });
  }

  addPoint(step, loss) {
    const point = { x: step, y: loss };
    const previous = this.history[this.history.length - 1];
    if (previous && previous.x === point.x) {
      previous.y = point.y;
    } else {
      this.history.push(point);
    }
    if (this.history.length > 1000) this.history.shift();
    if (this.chart) {
      this.chart.data.datasets[0].data = this.history;
      this.chart.update("none");
    }
  }

  reset() {
    this.history = [];
    if (this.chart) {
      this.chart.data.datasets[0].data = [];
      this.chart.update("none");
    }
  }

  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    this.canvas = null;
  }
}
