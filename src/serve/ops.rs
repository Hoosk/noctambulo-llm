//! Background operations (downloads, mixes, exports) with live progress and
//! cancellation, streamed to the UI over SSE.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Events published while operations run. Only `OpsRegistry` and `OpHandle`
/// construct them, so the server is the single source of truth.
#[derive(Debug, Clone, Serialize)]
pub enum OpEvent {
    Started {
        id: String,
        kind: String,
        label: String,
        started_unix: u64,
    },
    Progress {
        id: String,
        kind: String,
        phase: String,
        done: u64,
        total: u64,
        unit: String,
    },
    Done {
        id: String,
        kind: String,
        message: String,
        result: Option<serde_json::Value>,
    },
    Failed {
        id: String,
        kind: String,
        error: String,
    },
    Cancelled {
        id: String,
        kind: String,
    },
}

/// Snapshot of one operation, used to backfill the UI after reloads.
#[derive(Debug, Clone, Serialize)]
pub struct OpStatus {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub phase: String,
    pub done: u64,
    pub total: u64,
    pub unit: String,
    /// `running` | `done` | `failed` | `cancelled`
    pub status: String,
    pub message: Option<String>,
    pub error: Option<String>,
    pub result: Option<serde_json::Value>,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
}

/// Finished operations kept for the UI; older ones are pruned.
pub const MAX_OPS: usize = 60;

struct Inner {
    ops: HashMap<String, OpStatus>,
    cancel: HashMap<String, Arc<AtomicBool>>,
}

/// Registry of long-running operations. Unlike `JobRegistry` (single training
/// job), several ops can run at once.
pub struct OpsRegistry {
    inner: Mutex<Inner>,
    events: tokio::sync::broadcast::Sender<OpEvent>,
    seq: AtomicU64,
}

impl OpsRegistry {
    pub fn new() -> Self {
        let (events, _) = tokio::sync::broadcast::channel(256);
        Self {
            inner: Mutex::new(Inner {
                ops: HashMap::new(),
                cancel: HashMap::new(),
            }),
            events,
            seq: AtomicU64::new(1),
        }
    }

    /// Register and announce a new operation.
    pub fn start(self: &Arc<Self>, kind: &str, label: &str) -> OpHandle {
        let id = format!("op-{}", self.seq.fetch_add(1, Ordering::Relaxed));
        let started = now_unix();
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut inner = self.inner.lock().unwrap();
            prune(&mut inner.ops);
            inner.ops.insert(
                id.clone(),
                OpStatus {
                    id: id.clone(),
                    kind: kind.into(),
                    label: label.into(),
                    phase: "starting".into(),
                    done: 0,
                    total: 0,
                    unit: String::new(),
                    status: "running".into(),
                    message: None,
                    error: None,
                    result: None,
                    started_unix: started,
                    finished_unix: None,
                },
            );
            inner.cancel.insert(id.clone(), cancel.clone());
        }
        self.emit(OpEvent::Started {
            id: id.clone(),
            kind: kind.into(),
            label: label.into(),
            started_unix: started,
        });
        OpHandle {
            id,
            kind: kind.into(),
            cancel,
            registry: self.clone(),
        }
    }

    pub fn snapshot(&self) -> Vec<OpStatus> {
        let inner = self.inner.lock().unwrap();
        let mut out: Vec<OpStatus> = inner.ops.values().cloned().collect();
        out.sort_by(|a, b| {
            b.started_unix
                .cmp(&a.started_unix)
                .then_with(|| b.id.cmp(&a.id))
        });
        out
    }

    /// Request cancellation of a running operation. Returns false when the id
    /// is unknown or already finished.
    pub fn cancel(&self, id: &str) -> bool {
        let inner = self.inner.lock().unwrap();
        match (inner.ops.get(id), inner.cancel.get(id)) {
            (Some(op), Some(flag)) if op.status == "running" => {
                flag.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<OpEvent> {
        self.events.subscribe()
    }

    fn emit(&self, ev: OpEvent) {
        let _ = self.events.send(ev);
    }

    fn update(&self, id: &str, f: impl FnOnce(&mut OpStatus)) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(op) = inner.ops.get_mut(id) {
            f(op);
        }
    }
}

impl Default for OpsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn prune(ops: &mut HashMap<String, OpStatus>) {
    if ops.len() < MAX_OPS {
        return;
    }
    let mut finished: Vec<(u64, String)> = ops
        .values()
        .filter(|o| o.status != "running")
        .map(|o| (o.started_unix, o.id.clone()))
        .collect();
    finished.sort_unstable();
    let excess = ops.len().saturating_sub(MAX_OPS / 2);
    for (_, id) in finished.into_iter().take(excess) {
        ops.remove(&id);
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Handle used by the worker thread to report progress and finish the op.
#[derive(Clone)]
pub struct OpHandle {
    id: String,
    kind: String,
    cancel: Arc<AtomicBool>,
    registry: Arc<OpsRegistry>,
}

impl OpHandle {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn progress(&self, phase: &str, done: u64, total: u64, unit: &str) {
        if self.is_cancelled() {
            return;
        }
        self.registry.update(&self.id, |op| {
            op.phase = phase.into();
            op.done = done;
            op.total = total;
            op.unit = unit.into();
        });
        self.registry.emit(OpEvent::Progress {
            id: self.id.clone(),
            kind: self.kind.clone(),
            phase: phase.into(),
            done,
            total,
            unit: unit.into(),
        });
    }

    pub fn finish(&self, message: &str, result: Option<serde_json::Value>) {
        self.registry.update(&self.id, |op| {
            op.status = "done".into();
            op.phase = "done".into();
            op.message = Some(message.into());
            op.result = result.clone();
            op.finished_unix = Some(now_unix());
            if op.total > 0 {
                op.done = op.total;
            }
        });
        self.registry.emit(OpEvent::Done {
            id: self.id.clone(),
            kind: self.kind.clone(),
            message: message.into(),
            result,
        });
    }

    pub fn fail(&self, error: &str) {
        self.registry.update(&self.id, |op| {
            op.status = "failed".into();
            op.phase = "failed".into();
            op.error = Some(error.into());
            op.finished_unix = Some(now_unix());
        });
        self.registry.emit(OpEvent::Failed {
            id: self.id.clone(),
            kind: self.kind.clone(),
            error: error.into(),
        });
    }

    pub fn cancelled(&self) {
        self.registry.update(&self.id, |op| {
            op.status = "cancelled".into();
            op.phase = "cancelled".into();
            op.message = Some("cancelled".into());
            op.finished_unix = Some(now_unix());
        });
        self.registry.emit(OpEvent::Cancelled {
            id: self.id.clone(),
            kind: self.kind.clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_progress_finish_lifecycle() {
        let reg = Arc::new(OpsRegistry::new());
        let op = reg.start("download", "TinyStories");
        assert_eq!(op.id(), "op-1");
        assert!(!op.is_cancelled());

        op.progress("downloading", 50, 100, "bytes");
        let snap = reg.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].status, "running");
        assert_eq!((snap[0].done, snap[0].total), (50, 100));

        op.finish("ok", Some(serde_json::json!({ "path": "x" })));
        let snap = reg.snapshot();
        assert_eq!(snap[0].status, "done");
        assert_eq!(snap[0].done, 100);
        assert!(snap[0].finished_unix.is_some());
        assert_eq!(snap[0].result.as_ref().unwrap()["path"], "x");
    }

    #[test]
    fn test_cancel_only_while_running() {
        let reg = Arc::new(OpsRegistry::new());
        let op = reg.start("combine", "mix");
        assert!(reg.cancel(op.id()));
        assert!(op.is_cancelled());
        // Cancellation is idempotent while the worker is still winding down.
        assert!(reg.cancel(op.id()));
        op.cancelled();
        // A finished operation can no longer be cancelled.
        assert!(!reg.cancel(op.id()));
    }

    #[test]
    fn test_fail_records_error() {
        let reg = Arc::new(OpsRegistry::new());
        let op = reg.start("export", "gguf");
        op.fail("boom");
        let snap = reg.snapshot();
        assert_eq!(snap[0].status, "failed");
        assert_eq!(snap[0].error.as_deref(), Some("boom"));
    }

    #[test]
    fn test_prune_keeps_room_for_new_ops() {
        let reg = Arc::new(OpsRegistry::new());
        for i in 0..(MAX_OPS + 10) {
            let op = reg.start("export", &format!("op{i}"));
            op.finish("ok", None);
        }
        assert!(reg.snapshot().len() <= MAX_OPS);
    }
}
