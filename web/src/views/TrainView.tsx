import { ViewHeader } from "../components/ui";
import { useAppState } from "../state/appContext";
import { TrainForm } from "./train/TrainForm";
import { TrainMetrics } from "./train/TrainMetrics";
import { RunHistoryCard } from "./train/RunHistoryCard";

export function TrainView() {
  const { job } = useAppState();
  const active = job.status === "running";

  // While a job runs the wizard is hidden and the live metrics take the full
  // width (pause/stop live there); when idle only the wizard and the run
  // history are shown, with a compact summary of the last run.
  return (
    <div className="space-y-6 min-w-0">
      <ViewHeader
        title="Train"
        subtitle="Configure a run, watch the loss curve, and keep every checkpoint catalogued."
      />
      {active ? (
        <div className="stagger-children space-y-6">
          <TrainMetrics />
          <RunHistoryCard />
        </div>
      ) : (
        <div className="stagger-children grid min-w-0 grid-cols-1 xl:grid-cols-[minmax(0,600px)_minmax(0,1fr)] gap-6 items-start">
          <TrainForm />
          <RunHistoryCard />
        </div>
      )}
    </div>
  );
}
