import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DeviceInfo, Telemetry } from "../api/types";
import { Icon, type IconName } from "../components/Icon";
import { Badge, ViewHeader, type Tone } from "../components/ui";
import { fmtBytes, fmtDur } from "../lib/format";

const TYPE_INFO: Record<string, { tone: Tone; icon: IconName }> = {
  "Discrete GPU": { tone: "violet", icon: "zap" },
  "Integrated GPU": { tone: "sky", icon: "cpu" },
  CPU: { tone: "slate", icon: "cpu" },
  Auto: { tone: "emerald", icon: "sparkles" },
};

const CHIPS: Record<Tone, string> = {
  slate: "bg-slate-700/40 text-slate-300",
  violet: "bg-violet-500/15 text-violet-300",
  fuchsia: "bg-fuchsia-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
};

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 text-xs py-1.5">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono text-right tnum">{value}</span>
    </div>
  );
}

function DeviceCard({ device }: { device: DeviceInfo }) {
  const info = TYPE_INFO[device.device_type] ?? { tone: "slate" as Tone, icon: "info" as IconName };
  return (
    <li className="lift glass-card rounded-2xl p-5">
      <div className="flex items-center justify-between gap-3 mb-3">
        <span className="text-sm font-medium text-white flex items-center gap-2.5 min-w-0">
          <span className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${CHIPS[info.tone]}`}>
            <Icon name={info.icon} className="w-4 h-4" />
          </span>
          <span className="truncate">{device.name}</span>
        </span>
        <Badge tone={info.tone}>{device.device_type}</Badge>
      </div>
      <div className="border-t border-slate-800/70 pt-2">
        <InfoRow label="Backend" value={device.backend} />
        <InfoRow label="Driver" value={device.driver || "—"} />
        <InfoRow label="Vendor ID" value={device.vendor ? `0x${device.vendor.toString(16)}` : "—"} />
        <InfoRow label="Device ID" value={device.device ? `0x${device.device.toString(16)}` : "—"} />
        <InfoRow label="VRAM approx." value={device.vram_bytes ? fmtBytes(device.vram_bytes) : "—"} />
      </div>
    </li>
  );
}

function Meter({
  label,
  detail,
  pct,
  tone,
}: {
  label: string;
  detail: string;
  pct: number;
  tone: "violet" | "sky" | "emerald";
}) {
  const colors = {
    violet: "from-violet-500 to-fuchsia-500",
    sky: "from-sky-500 to-cyan-400",
    emerald: "from-emerald-500 to-teal-400",
  };
  return (
    <div>
      <div className="flex justify-between gap-3 text-xs mb-1">
        <span className="text-slate-500 truncate">{label}</span>
        <span className="text-slate-400 tnum font-mono shrink-0">{detail}</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${colors[tone]} transition-all duration-300`}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  );
}

const pctOf = (part: number, total: number) => (total > 0 ? (part / total) * 100 : 0);

function TelemetryCard() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await api.getTelemetry();
        if (!cancelled) {
          setTelemetry(res);
          setError("");
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "unknown");
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  if (error) return <p className="text-sm text-rose-400 mb-6">Telemetry error: {error}</p>;
  if (!telemetry) return null;

  const { memory, process, gpus, notes } = telemetry;
  const memUsed = memory.total_bytes - memory.available_bytes;

  return (
    <div className="glass-card rounded-2xl p-5 mb-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Live telemetry</h3>
        <span className="text-xs text-slate-500 tnum font-mono">up {fmtDur(process.uptime_secs)}</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4">
        <Meter
          label="System memory"
          detail={`${fmtBytes(memUsed)} / ${fmtBytes(memory.total_bytes)}`}
          pct={memory.used_pct}
          tone="violet"
        />
        <Meter
          label={`Process memory (${process.threads} threads)`}
          detail={fmtBytes(process.rss_bytes)}
          pct={pctOf(process.rss_bytes, memory.total_bytes)}
          tone="emerald"
        />
        <div>
          <div className="flex justify-between gap-3 text-xs mb-1">
            <span className="text-slate-500">Process CPU</span>
            <span className="text-slate-400 tnum font-mono">
              {process.cpu_pct != null ? `${process.cpu_pct.toFixed(1)}% of one core` : "sampling…"}
            </span>
          </div>
        </div>
        {gpus.map((gpu) => (
          <Meter
            key={`${gpu.source}-${gpu.index}`}
            label={`${gpu.name} VRAM (${gpu.source})`}
            detail={`${fmtBytes(gpu.used_bytes)} / ${fmtBytes(gpu.total_bytes)}`}
            pct={pctOf(gpu.used_bytes, gpu.total_bytes)}
            tone="sky"
          />
        ))}
      </div>
      {gpus.length === 0
        ? notes.map((note) => (
            <p key={note} className="text-xs text-slate-500 mt-4">
              {note}
            </p>
          ))
        : null}
    </div>
  );
}

export function DevicesView() {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getDevices()
      .then((list) => {
        if (!cancelled) {
          setDevices(list);
          setLoaded(true);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "unknown");
          setLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <>
      <ViewHeader title="GPU Devices" subtitle="WGPU adapters detected on the server host." />
      <TelemetryCard />
      {error ? <p className="text-sm text-rose-400 mb-4">Error: {error}</p> : null}
      {loaded && devices.length === 0 && !error ? (
        <div className="p-10 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500">
          No adapters detected.
        </div>
      ) : null}
      <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {devices.map((d) => (
          <DeviceCard key={`${d.backend}-${d.vendor}-${d.device}-${d.name}`} device={d} />
        ))}
      </ul>
    </>
  );
}
