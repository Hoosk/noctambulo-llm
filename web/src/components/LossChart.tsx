import { lazy, Suspense, useMemo } from "react";
import type { ChartData, ChartOptions, ScriptableContext } from "chart.js";
import type { LossPoint } from "../api/types";

const FONT = "'JetBrains Mono', ui-monospace, monospace";
const VIOLET = "139, 92, 246";
const FUCHSIA = "217, 70, 239";

const strokeGradient = (ctx: ScriptableContext<"line">) => {
  const { chartArea, ctx: g2d } = ctx.chart;
  if (!chartArea) return `rgb(${VIOLET})`;
  const gradient = g2d.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
  gradient.addColorStop(0, `rgb(${VIOLET})`);
  gradient.addColorStop(1, `rgb(${FUCHSIA})`);
  return gradient;
};

const fillGradient = (ctx: ScriptableContext<"line">) => {
  const { chartArea, ctx: g2d } = ctx.chart;
  if (!chartArea) return `rgba(${VIOLET}, 0.14)`;
  const gradient = g2d.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  gradient.addColorStop(0, `rgba(${VIOLET}, 0.26)`);
  gradient.addColorStop(1, `rgba(${VIOLET}, 0)`);
  return gradient;
};

const OPTIONS: ChartOptions<"line"> = {
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
    },
  },
  scales: {
    x: {
      type: "linear",
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
};

// Chart.js and its React wrapper are heavy and only needed once a run produces
// metrics, so both live behind a lazy boundary: the chunk loads when the chart
// first renders, not with the app.
const LazyLine = lazy(async () => {
  const [{ Line }, chart] = await Promise.all([import("react-chartjs-2"), import("chart.js")]);
  chart.Chart.register(
    chart.LineController,
    chart.LineElement,
    chart.PointElement,
    chart.LinearScale,
    chart.Filler,
    chart.Tooltip,
  );
  return { default: Line };
});

export function LossChart({ points }: { points: LossPoint[] }) {
  const data = useMemo<ChartData<"line", LossPoint[]>>(
    () => ({
      datasets: [
        {
          label: "Loss",
          data: points,
          borderColor: strokeGradient,
          backgroundColor: fillGradient,
          fill: true,
          tension: 0.3,
          pointRadius: (ctx) => (points.length > 1 && ctx.dataIndex === points.length - 1 ? 3 : 0),
          pointBackgroundColor: "#a78bfa",
          pointBorderColor: "#0f172a",
          pointBorderWidth: 1.5,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "#c4b5fd",
          borderWidth: 2,
        },
      ],
    }),
    [points],
  );

  return (
    <div className="relative h-64 w-full overflow-hidden">
      <Suspense fallback={<div className="h-full w-full rounded-xl bg-slate-950/40" />}>
        <LazyLine data={data} options={OPTIONS} />
      </Suspense>
    </div>
  );
}
