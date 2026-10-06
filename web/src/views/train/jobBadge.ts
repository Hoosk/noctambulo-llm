import type { JobInfo } from "../../state/jobModel";
import type { Tone } from "../../components/ui";

export interface BadgeInfo {
  tone: Tone;
  label: string;
  pulse?: boolean;
}

export function jobBadge(job: JobInfo): BadgeInfo {
  if (job.status === "running") {
    if (job.paused) return { tone: "amber", label: "paused" };
    if (job.stopping) return { tone: "amber", label: "stopping" };
    return { tone: "violet", label: "training", pulse: true };
  }
  if (job.status === "done") return { tone: "emerald", label: "completed" };
  if (job.status === "stopped") return { tone: "amber", label: "stopped" };
  if (job.status === "error") return { tone: "rose", label: "error" };
  return { tone: "slate", label: "idle" };
}
