import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetInfo } from "../api/types";
import { Button } from "../components/Button";
import { ViewHeader } from "../components/ui";
import { fmtBytes } from "../lib/format";
import { useModal } from "../state/modalContext";
import { useToast } from "../state/toastContext";
import { CombineCard } from "./datasets/CombineCard";
import { DatasetList } from "./datasets/DatasetList";
import { DownloadCard } from "./datasets/DownloadCard";
import { SearchCard } from "./datasets/SearchCard";

export function DatasetsView() {
  const toast = useToast();
  const modal = useModal();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loaded, setLoaded] = useState(false);

  const applyList = useCallback((list: DatasetInfo[]) => {
    setDatasets(list);
    setLoaded(true);
    setSelected((prev) => {
      const names = new Set(list.map((d) => d.name));
      return new Set([...prev].filter((name) => names.has(name)));
    });
  }, []);

  const reportError = useCallback(
    (e: unknown) => {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    },
    [toast],
  );

  const load = useCallback(async () => {
    try {
      applyList(await api.getDatasets());
    } catch (e) {
      reportError(e);
    }
  }, [applyList, reportError]);

  useEffect(() => {
    let cancelled = false;
    api
      .getDatasets()
      .then((list) => {
        if (!cancelled) applyList(list);
      })
      .catch((e: unknown) => {
        if (!cancelled) reportError(e);
      });
    return () => {
      cancelled = true;
    };
  }, [applyList, reportError]);

  const toggle = (name: string, checked: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (checked) next.add(name);
      else next.delete(name);
      return next;
    });
  };

  const toggleAll = (checked: boolean) => {
    setSelected(checked ? new Set(datasets.map((d) => d.name)) : new Set());
  };

  const showInfo = async (dataset: DatasetInfo) => {
    try {
      const info = await api.getDatasetInfo(dataset.name);
      const lines = [
        `Size: ${fmtBytes(info.size_bytes)}`,
        `Type: ${info.ext}`,
        info.lines != null ? `Lines: ${info.lines.toLocaleString("en-US")}` : "",
      ]
        .filter(Boolean)
        .join("\n");
      await modal.alert({ title: info.name, message: lines });
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const rename = async (dataset: DatasetInfo) => {
    const to = await modal.prompt({
      title: "Rename dataset",
      label: "New name",
      value: dataset.name,
    });
    if (!to || to === dataset.name) return;
    try {
      await api.renameDataset(dataset.name, to);
      toast.success(`Renamed to ${to}`);
      void load();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const remove = async (dataset: DatasetInfo) => {
    const ok = await modal.confirm({
      title: "Delete dataset?",
      message: dataset.name,
      confirmText: "Delete",
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteDataset(dataset.name);
      toast.success(`${dataset.name} deleted`);
      setSelected((prev) => {
        const next = new Set(prev);
        next.delete(dataset.name);
        return next;
      });
      void load();
    } catch (e) {
      toast.error(`Error: ${e instanceof Error ? e.message : "unknown"}`);
    }
  };

  const selectedNames = [...selected].sort();

  return (
    <>
      <ViewHeader
        title="Datasets"
        subtitle="Download, organize and combine corpora."
        actions={
          <Button icon="refresh" onClick={() => void load()}>
            Refresh
          </Button>
        }
      />
      <DownloadCard onDownloaded={() => void load()} />
      <div className="h-6" />
      <SearchCard onDownloaded={() => void load()} />
      {loaded && datasets.length === 0 ? (
        <div className="mt-6 p-8 rounded-2xl border border-dashed border-slate-700/70 text-center text-slate-500">
          No local datasets. Download one or drop .txt/.bin into datasets/.
        </div>
      ) : null}
      {datasets.length > 0 ? (
        <div className="mt-6">
          <DatasetList
            datasets={datasets}
            selected={selected}
            onToggle={toggle}
            onToggleAll={toggleAll}
            onInfo={(d) => void showInfo(d)}
            onRename={(d) => void rename(d)}
            onDelete={(d) => void remove(d)}
          />
        </div>
      ) : null}
      {datasets.length > 0 ? (
        <CombineCard
          selectedNames={selectedNames}
          onDone={() => {
            setSelected(new Set());
            void load();
          }}
        />
      ) : null}
    </>
  );
}
