import { lazy, Suspense, useMemo } from "react";
import type { ChartData, ChartOptions } from "chart.js";
import type { LossPoint } from "../api/types";

const FONT = "'JetBrains Mono', ui-monospace, monospace";

export interface ChartSeries {
  label: string;
  points: LossPoint[];
  /** Hex color, e.g. `#8b5cf6`. */
  color: string;
  dashed?: boolean;
}

function withAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function options(showLegend: boolean): ChartOptions<"line"> {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        display: showLegend,
        position: "top",
        labels: {
          color: "#94a3b8",
          boxWidth: 12,
          boxHeight: 2,
          font: { family: FONT, size: 10 },
        },
      },
      tooltip: {
        backgroundColor: "rgba(2, 6, 23, 0.92)",
        borderColor: "rgba(139, 92, 246, 0.35)",
        borderWidth: 1,
        cornerRadius: 10,
        padding: 10,
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
}

// Chart.js and its React wrapper are heavy and only needed once data exists,
// so both live behind a lazy boundary: the chunk loads with the first chart.
const LazyLine = lazy(async () => {
  const [{ Line }, chart] = await Promise.all([import("react-chartjs-2"), import("chart.js")]);
  chart.Chart.register(
    chart.LineController,
    chart.LineElement,
    chart.PointElement,
    chart.LinearScale,
    chart.Filler,
    chart.Tooltip,
    chart.Legend,
  );
  return { default: Line };
});

/** Multi-series line chart (train/val loss, run comparisons). */
export function LineChart({
  series,
  height = 256,
}: {
  series: ChartSeries[];
  height?: number;
}) {
  const data = useMemo<ChartData<"line", LossPoint[]>>(
    () => ({
      datasets: series.map((s) => ({
        label: s.label,
        data: s.points,
        borderColor: s.color,
        backgroundColor: series.length === 1 ? withAlpha(s.color, 0.12) : "transparent",
        fill: series.length === 1,
        tension: 0.3,
        borderDash: s.dashed ? [6, 4] : undefined,
        pointRadius: (ctx) =>
          s.points.length > 1 && ctx.dataIndex === s.points.length - 1 ? 3 : 0,
        pointBackgroundColor: s.color,
        pointBorderColor: "#0f172a",
        pointBorderWidth: 1.5,
        pointHoverRadius: 4,
        borderWidth: 2,
      })),
    }),
    [series],
  );

  return (
    <div className="relative w-full overflow-hidden" style={{ height }}>
      <Suspense fallback={<div className="h-full w-full rounded-xl bg-slate-950/40" />}>
        <LazyLine data={data} options={options(series.length > 1)} />
      </Suspense>
    </div>
  );
}
