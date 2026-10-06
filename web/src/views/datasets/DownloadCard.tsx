import { useState } from "react";
import { api } from "../../api/client";
import type { RepoFiles } from "../../api/types";
import { Button } from "../../components/Button";
import { OpProgress } from "../../components/OpProgress";
import { Field, Input } from "../../components/ui";
import { useAppState } from "../../state/appContext";
import { useToast } from "../../state/toastContext";
import { useOp, useOpOutcome } from "../../state/useOp";
import { FilePicker } from "./FilePicker";

type NoticeTone = "muted" | "ok" | "error";

const NOTICE_CLASSES: Record<NoticeTone, string> = {
  muted: "text-slate-400",
  ok: "text-emerald-300",
  error: "text-rose-400",
};

export function DownloadCard({ onDownloaded }: { onDownloaded: () => void }) {
  const toast = useToast();
  const { cancelOp } = useAppState();
  const [repoId, setRepoId] = useState("");
  const [filename, setFilename] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [repoFiles, setRepoFiles] = useState<RepoFiles | null>(null);
  const [listing, setListing] = useState(false);
  const [notice, setNotice] = useState<{ text: string; tone: NoticeTone }>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);
  const [opId, setOpId] = useState<string | null>(null);
  const op = useOp(opId);
  const running = op?.status === "running";

  useOpOutcome(opId, (o) => {
    if (o.status === "done") {
      toast.success(o.message ?? "Download complete");
      onDownloaded();
    } else if (o.status === "cancelled") {
      toast.info("Download cancelled");
    } else {
      toast.error(`Download failed: ${o.error ?? "unknown"}`);
    }
    setOpId(null);
  });

  // Offer the repo's real files once the user leaves the repo field; if the
  // listing fails the free-text filename still works.
  const loadFiles = async () => {
    const repo = repoId.trim();
    if (!repo || listing) return;
    setListing(true);
    try {
      const res = await api.getRepoFiles(repo);
      setRepoFiles(res.files.length > 0 ? res : null);
      setSelected(res.default ? [res.default] : []);
    } catch {
      setRepoFiles(null);
    } finally {
      setListing(false);
    }
  };

  const download = async () => {
    const repo = repoId.trim();
    if (!repo || busy || running) {
      if (!repo) setNotice({ text: "Enter a Hugging Face repo.", tone: "error" });
      return;
    }
    setBusy(true);
    setNotice({ text: "Starting download…", tone: "muted" });
    try {
      const files = repoFiles ? selected : filename.trim() ? [filename.trim()] : [];
      const res = await api.downloadDataset(repo, files);
      setOpId(res.opId);
    } catch (e) {
      setNotice({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-5">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-4">
        Download from Hugging Face
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Repo ID">
          <Input
            placeholder="HuggingFaceTB/cosmopedia-100k"
            value={repoId}
            onChange={(e) => {
              setRepoId(e.target.value);
              setRepoFiles(null);
              setSelected([]);
            }}
            onBlur={() => void loadFiles()}
          />
        </Field>
        <Field label={repoFiles ? "File" : "File (optional)"}>
          {repoFiles ? (
            <FilePicker files={repoFiles.files} selected={selected} onChange={setSelected} />
          ) : (
            <Input
              placeholder={listing ? "loading files…" : "auto · best training file"}
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
            />
          )}
        </Field>
      </div>
      <p className="text-[11px] text-slate-500 mt-2">
        Parquet files are converted to <span className="font-mono">_text.txt</span> automatically.
      </p>
      <div className="mt-4">
        <Button
          variant="primary"
          icon="download"
          disabled={busy || running}
          onClick={() => void download()}
        >
          {running ? "Downloading…" : busy ? "Starting…" : "Download"}
        </Button>
      </div>
      {op && running ? (
        <OpProgress op={op} onCancel={() => cancelOp(op.id)} />
      ) : (
        <p className={`text-xs mt-3 ${NOTICE_CLASSES[notice.tone]}`}>{notice.text}</p>
      )}
    </div>
  );
}
