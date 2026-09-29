import { createContext, useContext } from "react";
import type { OpStatus } from "../api/types";
import type { AuthState, Connection, JobInfo } from "./jobModel";

export interface AppStateValue {
  authState: AuthState;
  submitToken: (token: string) => Promise<boolean>;
  logout: () => void;
  conn: Connection;
  job: JobInfo;
  /** Background operations (downloads, mixes, exports) with live progress. */
  ops: OpStatus[];
  cancelOp: (id: string) => void;
  /** Model dir chosen in the Models view; shared with Generate/Export. */
  activeModel: string;
  setActiveModel: (dir: string) => void;
}

export const AppStateContext = createContext<AppStateValue | null>(null);

export function useAppState(): AppStateValue {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used inside <AppStateProvider>");
  return ctx;
}
