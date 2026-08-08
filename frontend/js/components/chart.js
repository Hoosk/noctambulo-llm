// Lightweight wrapper around Chart.js for the loss curve.

import { el } from "../core/dom.js";

export default class LossChart {
  constructor() {
    this.history = []; // {step, loss}
    this.chart = null;
    this.container = null;
  }

  build() {
    this.container = el("div", { class: "relative h-64" });
    this.container.innerHTML = `<canvas></canvas>`;
    const canvas = this.container.querySelector("canvas");
    const ctx = canvas.getContext("2d");

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Loss",
            data: [],
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139,92,246,0.08)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
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

    return this.container;
  }

  addPoint(step, loss) {
    this.history.push({ step, loss });
    if (this.history.length > 1000) this.history.shift();
    if (!this.chart) return;
    this.chart.data.datasets[0].data = this.history;
    this.chart.update("none");
  }

  reset() {
    this.history = [];
    if (this.chart) {
      this.chart.data.datasets[0].data = [];
      this.chart.update("none");
    }
  }
}
