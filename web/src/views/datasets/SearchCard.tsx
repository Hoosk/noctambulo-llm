import { useState } from "react";
import { api } from "../../api/client";
import type { DatasetHit, RepoFiles } from "../../api/types";
import { Button } from "../../components/Button";
import { Icon } from "../../components/Icon";
import { Badge, Input } from "../../components/ui";
import { useToast } from "../../state/toastContext";
import { FilePicker } from "./FilePicker";

type NoticeTone = "muted" | "ok" | "error";

const NOTICE_CLASSES: Record<NoticeTone, string> = {
  muted: "text-slate-400",
  ok: "text-emerald-300",
  error: "text-rose-400",
};

function SearchRow({ hit, onDownloaded }: { hit: DatasetHit; onDownloaded: () => void }) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const [files, setFiles] = useState<RepoFiles | null>(null);
  const [pick, setPick] = useState("");
  const language = hit.tags[0]?.replace("language:", "");

  const download = async (filename: string) => {
    if (busy) return;
    setBusy(true);
    try {
      const res = await api.downloadDataset(hit.id, filename);
      toast.success(`Downloading ${hit.id} · ${res.file}…`);
      setFiles(null);
      onDownloaded();
    } catch (e) {
      toast.error(`Download failed: ${e instanceof Error ? e.message : "unknown"}`);
    } finally {
      setBusy(false);
    }
  };

  // First click lists the real repo files and offers a picker; if the listing
  // fails the backend resolves the best training file on its own.
  const chooseFile = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const res = await api.getRepoFiles(hit.id);
      if (res.files.length === 0) {
        toast.error(`${hit.id} has no files to download.`);
        return;
      }
      setFiles(res);
      setPick(res.default ?? res.files[0]);
    } catch {
      setBusy(false);
      await download("");
      return;
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="lift rounded-xl bg-slate-950/70 border border-slate-800/70 px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-slate-200 font-mono truncate">{hit.id}</span>
            {language ? <Badge tone="sky">{language}</Badge> : null}
          </div>
          <div className="text-xs text-slate-500 mt-0.5 tnum">
            {hit.downloads.toLocaleString("en-US")} downloads · {hit.likes.toLocaleString("en-US")} likes
          </div>
        </div>
        {files === null ? (
          <Button
            variant="primary"
            icon="download"
            className="px-3 py-1.5 text-xs shrink-0"
            disabled={busy}
            onClick={() => void chooseFile()}
          >
            {busy ? "Loading…" : "Choose file"}
          </Button>
        ) : null}
      </div>
      {files ? (
        <div className="mt-3 space-y-2">
          <FilePicker files={files.files} value={pick} onChange={setPick} />
          <div className="flex items-center gap-2">
            <span className="flex-1 min-w-0 text-[11px] text-slate-500 font-mono truncate">{pick}</span>
            <Button
              variant="primary"
              className="px-3 py-1.5 text-xs shrink-0"
              disabled={busy}
              onClick={() => void download(pick)}
            >
              {busy ? "Starting…" : "Download"}
            </Button>
            <Button
              variant="ghost"
              className="px-3 py-1.5 text-xs shrink-0"
              onClick={() => setFiles(null)}
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function SearchCard({ onDownloaded }: { onDownloaded: () => void }) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<DatasetHit[] | null>(null);
  const [notice, setNotice] = useState<{ text: string; tone: NoticeTone }>({ text: "", tone: "muted" });
  const [busy, setBusy] = useState(false);

  const search = async () => {
    const q = query.trim();
    if (!q || busy) return;
    setBusy(true);
    setNotice({ text: "Searching…", tone: "muted" });
    try {
      const found = await api.searchDatasets(q);
      setHits(found);
      setNotice({ text: "", tone: "muted" });
    } catch (e) {
      setHits(null);
      setNotice({ text: `Error: ${e instanceof Error ? e.message : "unknown"}`, tone: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Search Hugging Face
        </h3>
      </div>
      <div className="flex gap-3 items-start">
        <div className="relative flex-1">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none">
            <Icon name="search" className="w-4 h-4" />
          </span>
          <Input
            className="pl-9"
            placeholder="Search datasets on Hugging Face (e.g. cosmopedia, stories, code)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void search();
            }}
          />
        </div>
        <Button variant="primary" disabled={busy} onClick={() => void search()}>
          Search
        </Button>
      </div>
      <p className={`text-xs mt-3 ${NOTICE_CLASSES[notice.tone]}`}>{notice.text}</p>
      <div className="mt-3 space-y-2">
        {hits !== null && hits.length === 0 ? (
          <p className="text-xs text-slate-500">No results.</p>
        ) : (
          (hits ?? []).map((hit) => (
            <SearchRow key={hit.id} hit={hit} onDownloaded={onDownloaded} />
          ))
        )}
      </div>
    </div>
  );
}
