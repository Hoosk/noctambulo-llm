import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DeviceInfo } from "../api/types";
import { Icon, type IconName } from "../components/Icon";
import { Badge, ViewHeader, type Tone } from "../components/ui";
import { fmtBytes } from "../lib/format";

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
