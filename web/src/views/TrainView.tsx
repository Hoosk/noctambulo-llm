import { TrainForm } from "./train/TrainForm";
import { TrainMetrics } from "./train/TrainMetrics";
import { RunHistoryCard } from "./train/RunHistoryCard";

export function TrainView() {
  return (
    <div className="grid min-w-0 grid-cols-1 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] gap-6 items-start">
      <TrainForm />
      <TrainMetrics />
      <div className="lg:col-span-2 w-full">
        <RunHistoryCard />
      </div>
    </div>
  );
}
