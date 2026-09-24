// Lightweight wrapper around Chart.js for the loss curve.
// Chart.js must be initialized only after the canvas is attached to the DOM,
// otherwise it measures a wrong (huge) container and breaks the layout.

import { el } from "../core/dom.js";

const FONT = "'JetBrains Mono', ui-monospace, monospace";
const VIOLET = "139, 92, 246";
const FUCHSIA = "217, 70, 239";

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

    // Gradient helpers resolve lazily: chartArea is unknown until first layout.
    const strokeGradient = (c) => {
      const { chartArea, ctx: g2d } = c.chart;
      if (!chartArea) return `rgb(${VIOLET})`;
      const g = g2d.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
      g.addColorStop(0, `rgb(${VIOLET})`);
      g.addColorStop(1, `rgb(${FUCHSIA})`);
      return g;
    };
    const fillGradient = (c) => {
      const { chartArea, ctx: g2d } = c.chart;
      if (!chartArea) return `rgba(${VIOLET}, 0.14)`;
      const g = g2d.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
      g.addColorStop(0, `rgba(${VIOLET}, 0.26)`);
      g.addColorStop(1, `rgba(${VIOLET}, 0)`);
      return g;
    };

    this.chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Loss",
            data: this.history,
            borderColor: strokeGradient,
            backgroundColor: fillGradient,
            fill: true,
            tension: 0.3,
            pointRadius: (c) =>
              this.history.length > 1 && c.dataIndex === this.history.length - 1 ? 3 : 0,
            pointBackgroundColor: "#a78bfa",
            pointBorderColor: "#0f172a",
            pointBorderWidth: 1.5,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: "#c4b5fd",
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
            backgroundColor: "rgba(2, 6, 23, 0.92)",
            borderColor: `rgba(${VIOLET}, 0.35)`,
            borderWidth: 1,
            cornerRadius: 10,
            padding: 10,
            displayColors: false,
            titleColor: "#e2e8f0",
            bodyColor: "#c4b5fd",
            titleFont: { family: FONT, size: 10 },
            bodyFont: { family: FONT, size: 12, weight: 600 },
            callbacks: {
              title: (items) => (items.length ? `step ${items[0].parsed.x}` : ""),
              label: (item) => `loss ${Number(item.parsed.y).toFixed(4)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(51, 65, 85, 0.15)" },
            border: { color: "rgba(51, 65, 85, 0.4)" },
            ticks: { color: "#64748b", maxTicksLimit: 10, font: { family: FONT, size: 10 } },
            title: { display: true, text: "step", color: "#475569", font: { size: 10 } },
          },
          y: {
            grid: { color: "rgba(51, 65, 85, 0.15)" },
            border: { color: "rgba(51, 65, 85, 0.4)" },
            ticks: { color: "#64748b", font: { family: FONT, size: 10 } },
            title: { display: true, text: "loss", color: "#475569", font: { size: 10 } },
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
    if (this.history.length > 2000) this.history.shift();
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

  // Repopulate the curve from persisted/backfilled points ({x, y}) without
  // wiping it on reloads. Call before the first live addPoint.
  setHistory(points) {
    if (!Array.isArray(points) || points.length === 0) return;
    this.history = points.slice();
    if (this.history.length > 2000) this.history.splice(0, this.history.length - 2000);
    if (this.chart) {
      this.chart.data.datasets[0].data = this.history;
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
