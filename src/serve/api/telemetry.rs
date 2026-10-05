//! Live system telemetry: process CPU/RSS, system memory and GPU memory
//! counters where the OS exposes them (nvidia-smi or DRM sysfs).

use std::time::Instant;

use axum::{Json, extract::State, http::HeaderMap};
use serde::Serialize;

use super::AppState;
use super::auth::check_token;
use super::error::ApiError;

#[derive(Serialize)]
pub(super) struct MemoryInfo {
    total_bytes: u64,
    available_bytes: u64,
    used_pct: f64,
}

#[derive(Serialize)]
pub(super) struct ProcessInfo {
    rss_bytes: u64,
    /// Percent of a single core; `None` on the first sample (no baseline yet).
    cpu_pct: Option<f64>,
    /// Percent of wall time the process kept the GPU busy (from DRM fdinfo).
    /// `None` when the driver exposes no engine counters or on the first sample.
    gpu_pct: Option<f64>,
    threads: usize,
    uptime_secs: u64,
}

#[derive(Serialize)]
pub(super) struct GpuMemory {
    index: u32,
    name: String,
    used_bytes: u64,
    total_bytes: u64,
    /// System-wide GPU utilization when the driver exposes it (`nvidia-smi`,
    /// AMD `gpu_busy_percent`); `None` otherwise (e.g. Intel i915 needs
    /// CAP_PERFMON, so the process GPU counter is reported instead).
    utilization_pct: Option<f64>,
    /// Where the numbers came from (`nvidia-smi` or `sysfs`).
    source: String,
}

#[derive(Serialize)]
pub(super) struct Telemetry {
    memory: MemoryInfo,
    process: ProcessInfo,
    gpus: Vec<GpuMemory>,
    notes: Vec<String>,
}

/// `(sampled_at, cpu_ticks)` of the previous call, for delta-based CPU%.
static CPU_SAMPLE: std::sync::Mutex<Option<(Instant, u64)>> = std::sync::Mutex::new(None);

/// `(sampled_at, engine_ns)` of the previous call, for delta-based process GPU%.
static GPU_SAMPLE: std::sync::Mutex<Option<(Instant, u64)>> = std::sync::Mutex::new(None);

fn clk_tck() -> u64 {
    unsafe { libc::sysconf(libc::_SC_CLK_TCK) as u64 }.max(1)
}

fn page_size() -> u64 {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 }.max(1)
}

fn parse_kb(field: &str) -> Option<u64> {
    field
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

/// `(MemTotal, MemAvailable)` in bytes from `/proc/meminfo`.
fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_kb(rest);
        }
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_kb(rest);
        }
    }
    Some((total?, available?))
}

/// `(cpu_ticks, rss_pages, starttime_ticks)` from `/proc/self/stat`.
/// `comm` may contain spaces and parentheses, so parsing starts at the last `)`.
fn parse_self_stat(text: &str) -> Option<(u64, u64, u64)> {
    let close = text.rfind(')')?;
    let parts: Vec<&str> = text[close + 1..].split_whitespace().collect();
    let utime: u64 = parts.get(11)?.parse().ok()?;
    let stime: u64 = parts.get(12)?.parse().ok()?;
    let starttime: u64 = parts.get(19)?.parse().ok()?;
    let rss_pages: u64 = parts.get(21)?.parse().ok()?;
    Some((utime + stime, rss_pages, starttime))
}

fn parse_threads(status: &str) -> Option<usize> {
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Threads:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn parse_uptime(text: &str) -> Option<f64> {
    text.split_whitespace().next()?.parse().ok()
}

fn parse_nvidia_smi(text: &str) -> Vec<GpuMemory> {
    text.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(str::trim).collect();
            if cols.len() < 5 {
                return None;
            }
            let index = cols[0].parse().ok()?;
            let utilization_pct = cols[2].parse::<f64>().ok();
            let used_mib: u64 = cols[3].parse().ok()?;
            let total_mib: u64 = cols[4].parse().ok()?;
            Some(GpuMemory {
                index,
                name: cols[1].to_string(),
                used_bytes: used_mib * 1024 * 1024,
                total_bytes: total_mib * 1024 * 1024,
                utilization_pct,
                source: "nvidia-smi".into(),
            })
        })
        .collect()
}

/// Max busy time across the DRM engines this process has open, from
/// `/proc/self/fdinfo/*` (`drm-engine-<name>: <ns> ns`). Works without root on
/// i915/xe/amdgpu and complements the system-wide counters.
fn process_gpu_busy_ns() -> Option<u64> {
    let entries = std::fs::read_dir("/proc/self/fdinfo").ok()?;
    let mut max_ns: Option<u64> = None;
    for entry in entries.flatten() {
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("drm-engine-") else {
                continue;
            };
            let Some((key, value)) = rest.split_once(':') else {
                continue;
            };
            if key.starts_with("capacity") {
                continue;
            }
            let Some(ns) = value
                .trim()
                .strip_suffix("ns")
                .and_then(|v| v.trim().parse::<u64>().ok())
            else {
                continue;
            };
            max_ns = Some(max_ns.map_or(ns, |m| m.max(ns)));
        }
    }
    max_ns
}

/// AMD (and some other drivers) expose VRAM counters through DRM sysfs.
fn sysfs_gpus() -> Vec<GpuMemory> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut index = 0u32;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = entry.path().join("device");
        let used = std::fs::read_to_string(dev.join("mem_info_vram_used"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        let total = std::fs::read_to_string(dev.join("mem_info_vram_total"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let (Some(used_bytes), Some(total_bytes)) = (used, total) {
            let utilization_pct = std::fs::read_to_string(dev.join("gpu_busy_percent"))
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok());
            out.push(GpuMemory {
                index,
                name,
                used_bytes,
                total_bytes,
                utilization_pct,
                source: "sysfs".into(),
            });
            index += 1;
        }
    }
    out
}

fn collect() -> Telemetry {
    let mut notes = Vec::new();

    let memory = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| parse_meminfo(&text))
        .map(|(total, available)| {
            let used = total.saturating_sub(available);
            MemoryInfo {
                total_bytes: total,
                available_bytes: available,
                used_pct: if total > 0 {
                    used as f64 / total as f64 * 100.0
                } else {
                    0.0
                },
            }
        })
        .unwrap_or(MemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
            used_pct: 0.0,
        });
    if memory.total_bytes == 0 {
        notes.push("System memory counters unavailable (/proc/meminfo missing).".into());
    }

    let hz = clk_tck();
    let parsed = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|text| parse_self_stat(&text));
    let (ticks, rss_pages, starttime) = parsed.unwrap_or((0, 0, 0));

    // CPU usage needs two samples: keep the previous one in a static.
    let cpu_pct = {
        let mut sample = CPU_SAMPLE.lock().unwrap();
        let previous = *sample;
        let now = Instant::now();
        let pct = previous.and_then(|(then, then_ticks)| {
            let dt = now.duration_since(then).as_secs_f64();
            if dt > 0.0 && ticks >= then_ticks {
                Some((ticks - then_ticks) as f64 / hz as f64 / dt * 100.0)
            } else {
                None
            }
        });
        *sample = Some((now, ticks));
        pct
    };

    // Process GPU usage needs two engine samples, like CPU%.
    let gpu_pct = {
        let mut sample = GPU_SAMPLE.lock().unwrap();
        let previous = *sample;
        let now = Instant::now();
        let busy_ns = process_gpu_busy_ns();
        let pct = previous.and_then(|(then, then_ns)| {
            let dt = now.duration_since(then).as_secs_f64();
            let now_ns = busy_ns?;
            if dt > 0.0 && now_ns >= then_ns {
                Some((now_ns - then_ns) as f64 / 1e9 / dt * 100.0)
            } else {
                None
            }
        });
        if let Some(ns) = busy_ns {
            *sample = Some((now, ns));
        }
        pct
    };

    let uptime_secs = std::fs::read_to_string("/proc/uptime")
        .ok()
        .and_then(|text| parse_uptime(&text))
        .map(|boot_secs| (boot_secs - starttime as f64 / hz as f64).max(0.0) as u64)
        .unwrap_or(0);
    let threads = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| parse_threads(&text))
        .unwrap_or(0);

    let mut gpus = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout)))
        .unwrap_or_default();
    if gpus.is_empty() {
        gpus = sysfs_gpus();
    }
    if gpus.is_empty() {
        notes.push(
            "No GPU memory counters exposed by the OS on this system; only process and system memory are reported."
                .into(),
        );
    } else if gpus.iter().all(|g| g.utilization_pct.is_none()) {
        notes.push(
            "System-wide GPU utilization is not exposed by this driver (Intel needs CAP_PERFMON); \
             'Process GPU' shows this server's own engine busy time."
                .into(),
        );
    }

    Telemetry {
        memory,
        process: ProcessInfo {
            rss_bytes: rss_pages * page_size(),
            cpu_pct,
            gpu_pct,
            threads,
            uptime_secs,
        },
        gpus,
        notes,
    }
}

pub(super) async fn telemetry(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Telemetry>, ApiError> {
    check_token(&state.server, &headers, None)?;
    tokio::task::spawn_blocking(collect)
        .await
        .map(Json)
        .map_err(|e| ApiError::internal(format!("telemetry task failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_meminfo() {
        let text = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    4096000 kB\n";
        assert_eq!(parse_meminfo(text), Some((16_384_000 * 1024, 4_096_000 * 1024)));
        assert_eq!(parse_meminfo("MemFree: 1 kB\n"), None);
    }

    #[test]
    fn test_parse_self_stat_handles_spaces_in_comm() {
        // Fields after the last ')': state + ppid..rss (see proc(5)).
        let rest = "R 1 1 1 0 -1 4194560 100 0 0 0 11 22 0 0 20 0 1 0 33 1000000 25";
        let line = format!("42 (kuro (test) name) {rest}");
        assert_eq!(parse_self_stat(&line), Some((33, 25, 33)));
    }

    #[test]
    fn test_parse_threads() {
        assert_eq!(parse_threads("Name:\tkuro\nThreads:\t17\n"), Some(17));
        assert_eq!(parse_threads("Name:\tkuro\n"), None);
    }

    #[test]
    fn test_parse_nvidia_smi_csv() {
        let text = "0, NVIDIA GeForce RTX 4090, 37, 1234, 24564\n1, NVIDIA A100, 0, 0, 81920\n";
        let gpus = parse_nvidia_smi(text);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(gpus[0].used_bytes, 1234 * 1024 * 1024);
        assert_eq!(gpus[0].utilization_pct, Some(37.0));
        assert_eq!(gpus[1].total_bytes, 81920 * 1024 * 1024);
        assert_eq!(gpus[0].source, "nvidia-smi");
    }
}
