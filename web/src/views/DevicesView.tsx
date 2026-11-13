import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { DeviceInfo, RemoteNodeInfo, SystemInfo, Telemetry } from "../api/types";
import { Button } from "../components/Button";
import { Icon, type IconName } from "../components/Icon";
import { Badge, Field, Input, ViewHeader, type Tone } from "../components/ui";
import { fmtBytes, fmtDur } from "../lib/format";
import { useToast } from "../state/toastContext";

const TYPE_INFO: Record<string, { tone: Tone; icon: IconName }> = {
  "Discrete GPU": { tone: "violet", icon: "zap" },
  "Integrated GPU": { tone: "sky", icon: "cpu" },
  CPU: { tone: "slate", icon: "cpu" },
  Auto: { tone: "emerald", icon: "sparkles" },
  Remote: { tone: "fuchsia", icon: "network" },
};

const CHIPS: Record<Tone, string> = {
  slate: "bg-slate-800/60 text-slate-300",
  violet: "bg-amber-500/15 text-amber-300",
  fuchsia: "bg-fuchsia-500/15 text-fuchsia-300",
  emerald: "bg-emerald-500/15 text-emerald-300",
  amber: "bg-amber-500/15 text-amber-300",
  rose: "bg-rose-500/15 text-rose-300",
  sky: "bg-sky-500/15 text-sky-300",
};

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 text-xs py-1.5">
      <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono text-right tnum">{value}</span>
    </div>
  );
}

function dtypes(device: DeviceInfo): string {
  return [device.supports_f32 && "f32", device.supports_f16 && "f16", device.supports_bf16 && "bf16"]
    .filter(Boolean)
    .join(" · ");
}

function DeviceCard({
  device,
  onDefault,
  onCopy,
}: {
  device: DeviceInfo;
  onDefault: (selector: string) => void;
  onCopy: (selector: string) => void;
}) {
  const info = TYPE_INFO[device.device_type] ?? { tone: "slate" as Tone, icon: "info" as IconName };
  const offline = device.status !== "online" || !device.selectable;
  return (
    <li className={`lift glass-card corner-tick rounded-xl p-5 ${offline ? "opacity-80" : ""}`}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <span className="text-sm font-medium text-ivory flex items-start gap-3 min-w-0">
          <span
            className={`w-9 h-9 rounded-md flex items-center justify-center shrink-0 border border-slate-700/60 ${CHIPS[info.tone]}`}
          >
            <Icon name={info.icon} className="w-4 h-4" />
          </span>
          <span className="min-w-0">
            <span className="block truncate">{device.name}</span>
            <span className="mt-0.5 block font-mono text-[9px] uppercase tracking-[0.18em] text-slate-500">
              {device.backend} · {device.device_type}
            </span>
          </span>
        </span>
        <div className="flex flex-col items-end gap-1">
          {device.default ? <Badge tone="emerald" pulse>default</Badge> : null}
          {device.remote ? <Badge tone="fuchsia">remote</Badge> : null}
        </div>
      </div>

      <div className="border-t border-slate-800/70 pt-2">
        <InfoRow label="Status" value={device.status} />
        <InfoRow label="Dtypes" value={dtypes(device) || "—"} />
        <InfoRow label="Driver" value={device.driver || "—"} />
        <InfoRow label="Vendor" value={device.vendor ? `0x${device.vendor.toString(16)}` : "—"} />
        <InfoRow label="Device" value={device.device ? `0x${device.device.toString(16)}` : "—"} />
        <InfoRow label="VRAM" value={device.vram_bytes ? fmtBytes(device.vram_bytes) : "—"} />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <code className="min-w-0 flex-1 truncate rounded-md border border-slate-800/80 bg-slate-950/70 px-2 py-1.5 font-mono text-[10px] text-amber-200/80">
          {device.selector}
        </code>
        <Button
          variant="ghost"
          icon="copy"
          className="px-2.5 py-1.5 text-[10px]"
          onClick={() => onCopy(device.selector)}
        >
          Copy
        </Button>
        <Button
          variant={device.default ? "success" : "primary"}
          className="px-2.5 py-1.5 text-[10px]"
          disabled={device.default || offline}
          onClick={() => onDefault(device.selector)}
        >
          {device.default ? "Default" : "Set default"}
        </Button>
      </div>
    </li>
  );
}

function FabricStrip({ devices }: { devices: DeviceInfo[] }) {
  const backends = [...new Set(devices.map((d) => d.backend))];
  const online = devices.filter((d) => d.status === "online").length;
  const defaults = devices.find((d) => d.default)?.selector ?? "auto";
  return (
    <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
      <div className="glass-card corner-tick rounded-lg px-4 py-3">
        <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-500">Fabric</div>
        <div className="mt-1 font-mono text-sm text-ivory">
          {online}/{devices.length} online
        </div>
        <div className="mt-1 text-[10px] text-slate-500">local + remote accelerators</div>
      </div>
      <div className="glass-card corner-tick rounded-lg px-4 py-3">
        <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-500">Backends</div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          {backends.length === 0 ? (
            <span className="text-xs text-slate-500">none</span>
          ) : (
            backends.map((b) => (
              <span
                key={b}
                className="rounded-[3px] border border-brass/30 bg-amber-500/10 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.08em] text-amber-200/80"
              >
                {b}
              </span>
            ))
          )}
        </div>
        <div className="mt-1 text-[10px] text-slate-500">compiled and reachable</div>
      </div>
      <div className="glass-card corner-tick rounded-lg px-4 py-3">
        <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-500">Default device</div>
        <div className="mt-1 truncate font-mono text-sm text-amber-200">{defaults}</div>
        <div className="mt-1 text-[10px] text-slate-500">used when a request omits one</div>
      </div>
    </div>
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
    violet: "from-amber-500 to-fuchsia-500",
    sky: "from-sky-600 to-sky-400",
    emerald: "from-emerald-600 to-emerald-400",
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

function SystemCard() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getSystem()
      .then((res) => {
        if (!cancelled) setSystem(res);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "unknown");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p className="text-sm text-rose-400 mb-6">System info error: {error}</p>;
  if (!system) {
    return (
      <div className="glass-card corner-tick rounded-xl p-5 mb-6">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
            Host system
          </h3>
          <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-slate-600">
            reading…
          </span>
        </div>
        <InfoRow label="CPU" value="—" />
        <InfoRow label="Cores" value="—" />
        <InfoRow label="Kernel" value="—" />
      </div>
    );
  }

  const ramUsed = system.used_memory_bytes;
  const cpuLabel = system.cpu_physical_cores
    ? `${system.cpu_physical_cores} physical / ${system.cpu_logical_cores} logical`
    : `${system.cpu_logical_cores} logical`;

  return (
    <div className="glass-card corner-tick rounded-xl p-5 mb-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          Host system
        </h3>
        <span className="text-xs text-slate-500 font-mono truncate">
          {system.hostname} · up {fmtDur(system.uptime_secs)}
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4">
        <div className="sm:col-span-2">
          <InfoRow label="CPU" value={system.cpu_brand || "unknown"} />
          <div className="mt-2 mb-1">
            <Meter
              label="CPU usage"
              detail={`${system.cpu_usage_pct.toFixed(1)}%`}
              pct={system.cpu_usage_pct}
              tone="violet"
            />
          </div>
          {system.cpu_per_core.length > 0 ? (
            <div className="mb-2">
              <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-slate-600 mb-1">
                Per-core
              </div>
              <div className="flex items-end gap-[3px] h-8">
                {system.cpu_per_core.map((usage, i) => (
                  <div
                    key={i}
                    title={`core ${i}: ${usage.toFixed(0)}%`}
                    className="flex-1 min-w-[3px] rounded-sm bg-gradient-to-t from-amber-600 to-amber-300 transition-all duration-300"
                    style={{ height: `${Math.max(4, Math.min(100, usage))}%` }}
                  />
                ))}
              </div>
            </div>
          ) : null}
          <InfoRow label="Cores" value={cpuLabel} />
          <InfoRow
            label="Max frequency"
            value={system.cpu_frequency_mhz > 0 ? `${system.cpu_frequency_mhz} MHz` : "—"}
          />
          <InfoRow
            label="Load average"
            value={system.load_average.map((v) => v.toFixed(2)).join(" · ")}
          />
          <InfoRow label="OS" value={`${system.os_name} ${system.os_version}`} />
          <InfoRow label="Kernel" value={system.kernel_version} />
          {system.accelerators.map((acc) => (
            <InfoRow
              key={`${acc.pci_id}-${acc.slot}`}
              label="Accelerator"
              value={`${acc.name}${acc.driver ? ` · ${acc.driver}` : ""}${acc.pci_id ? ` · ${acc.pci_id}` : ""}`}
            />
          ))}
        </div>
        <Meter
          label="System memory (RAM)"
          detail={`${fmtBytes(ramUsed)} / ${fmtBytes(system.total_memory_bytes)}`}
          pct={pctOf(ramUsed, system.total_memory_bytes)}
          tone="violet"
        />
        {system.total_swap_bytes > 0 ? (
          <Meter
            label="Swap"
            detail={`${fmtBytes(system.used_swap_bytes)} / ${fmtBytes(system.total_swap_bytes)}`}
            pct={pctOf(system.used_swap_bytes, system.total_swap_bytes)}
            tone="emerald"
          />
        ) : null}
        {system.disks.map((disk) => (
          <Meter
            key={`${disk.mount_point}-${disk.name}`}
            label={`Disk ${disk.mount_point}`}
            detail={`${fmtBytes(disk.total_bytes - disk.available_bytes)} / ${fmtBytes(disk.total_bytes)}`}
            pct={pctOf(disk.total_bytes - disk.available_bytes, disk.total_bytes)}
            tone="sky"
          />
        ))}
      </div>
    </div>
  );
}

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
    <div className="glass-card corner-tick rounded-xl p-5 mb-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          Live telemetry
        </h3>
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
        <div className="grid grid-cols-2 gap-x-8">
          <div>
            <div className="flex justify-between gap-3 text-xs mb-1">
              <span className="text-slate-500">Process CPU</span>
              <span className="text-slate-400 tnum font-mono">
                {process.cpu_pct != null ? `${process.cpu_pct.toFixed(1)}% of one core` : "sampling…"}
              </span>
            </div>
          </div>
          <div>
            <div className="flex justify-between gap-3 text-xs mb-1">
              <span className="text-slate-500">Process GPU</span>
              <span className="text-slate-400 tnum font-mono">
                {process.gpu_pct != null ? `${process.gpu_pct.toFixed(1)}% busy` : "—"}
              </span>
            </div>
          </div>
        </div>
        {gpus.map((gpu) => (
          <Meter
            key={`${gpu.source}-${gpu.index}`}
            label={`${gpu.name} VRAM (${gpu.source})`}
            detail={`${fmtBytes(gpu.used_bytes)} / ${fmtBytes(gpu.total_bytes)}${
              gpu.utilization_pct != null ? ` · ${gpu.utilization_pct.toFixed(0)}% util` : ""
            }`}
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

function RemotePanel({
  nodes,
  onChanged,
}: {
  nodes: RemoteNodeInfo[];
  onChanged: () => void;
}) {
  const toast = useToast();
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const add = async () => {
    const target = url.trim();
    if (!target || busy) return;
    setBusy(true);
    setError("");
    try {
      const res = await api.addRemoteNode({
        name: name.trim() || undefined,
        url: target,
        token: token.trim() || undefined,
      });
      toast.success(`${res.name} online · ${res.devices} device(s)`);
      setName("");
      setUrl("");
      setToken("");
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "unknown");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (node: RemoteNodeInfo) => {
    try {
      await api.removeRemoteNode(node.id);
      toast.success(`${node.name} removed`);
      onChanged();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  return (
    <div className="glass-card corner-tick rounded-xl p-5 mb-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
            Remote compute nodes
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            Attach accelerators from other machines over Burn Remote (WebSocket or Iroh).
          </p>
        </div>
        <Badge tone={nodes.length > 0 ? "fuchsia" : "slate"}>{nodes.length} attached</Badge>
      </div>

      {nodes.length > 0 ? (
        <ul className="mb-4 space-y-2">
          {nodes.map((node) => (
            <li
              key={node.id}
              className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-800/80 bg-slate-950/50 px-3 py-2"
            >
              <Icon name="server" className="w-4 h-4 text-fuchsia-300/80 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm text-ivory">{node.name}</div>
                <div className="truncate font-mono text-[10px] text-slate-500">{node.url}</div>
              </div>
              {node.has_token ? <Badge tone="amber">token</Badge> : null}
              <Button
                variant="ghost"
                icon="trash"
                className="px-2 py-1 text-[10px]"
                onClick={() => void remove(node)}
              >
                Remove
              </Button>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Field label="Name (optional)">
          <Input
            value={name}
            placeholder="studio workstation"
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <Field label="Remote URL">
          <Input
            value={url}
            placeholder="ws://192.168.1.20:3000"
            onChange={(e) => setUrl(e.target.value)}
          />
        </Field>
        <Field label="Token (optional)">
          <Input
            type="password"
            value={token}
            placeholder="remote token"
            onChange={(e) => setToken(e.target.value)}
          />
        </Field>
        <div className="flex items-end">
          <Button
            variant="primary"
            icon={busy ? "loader" : "plus"}
            className={`w-full ${busy ? "[&_svg]:animate-spin" : ""}`}
            disabled={busy || !url.trim()}
            onClick={() => void add()}
          >
            {busy ? "Connecting…" : "Attach node"}
          </Button>
        </div>
      </div>
      {error ? <p className="mt-3 text-xs text-rose-400">{error}</p> : null}
    </div>
  );
}

export function DevicesView() {
  const toast = useToast();
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [nodes, setNodes] = useState<RemoteNodeInfo[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [list, remotes] = await Promise.all([api.getDevices(), api.getRemoteNodes()]);
      setDevices(list);
      setNodes(remotes);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "unknown");
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const onDefault = async (selector: string) => {
    try {
      await api.setDefaultDevice(selector);
      toast.success(`Default device: ${selector}`);
      await load();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const onCopy = async (selector: string) => {
    try {
      await navigator.clipboard.writeText(selector);
      toast.success(`Copied ${selector}`);
    } catch {
      toast.error("Clipboard unavailable");
    }
  };

  return (
    <>
      <ViewHeader
        title="Devices"
        subtitle="Compute fabric, local accelerators and attached remote hosts."
        actions={
          <Button icon="refresh" onClick={() => void load()}>
            Refresh
          </Button>
        }
      />
      <FabricStrip devices={devices} />
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          Accelerators
        </h3>
        <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-slate-600">
          {devices.length} selectable · click a selector to copy
        </span>
      </div>
      {error ? <p className="text-sm text-rose-400 mb-4">Error: {error}</p> : null}
      {loaded && devices.length === 0 && !error ? (
        <div className="p-10 rounded-xl border border-dashed border-slate-700/70 text-center text-slate-500">
          No adapters detected.
        </div>
      ) : null}
      <ul className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-4 stagger-children">
        {devices.map((d) => (
          <DeviceCard
            key={`${d.backend}-${d.selector}`}
            device={d}
            onDefault={(selector) => void onDefault(selector)}
            onCopy={(selector) => void onCopy(selector)}
          />
        ))}
      </ul>
      <RemotePanel nodes={nodes} onChanged={() => void load()} />
      <SystemCard />
      <TelemetryCard />
    </>
  );
}
