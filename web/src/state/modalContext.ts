import { createContext, useContext } from "react";

export interface ConfirmOptions {
  title?: string;
  message?: string;
  confirmText?: string;
  danger?: boolean;
}

export interface PromptOptions {
  title?: string;
  label?: string;
  value?: string;
  placeholder?: string;
}

export type DialogState =
  | { kind: "confirm"; options: ConfirmOptions; resolve: (v: boolean) => void }
  | { kind: "prompt"; options: PromptOptions; resolve: (v: string | null) => void }
  | { kind: "alert"; options: ConfirmOptions; resolve: () => void };

export interface ModalApi {
  confirm: (options?: ConfirmOptions) => Promise<boolean>;
  prompt: (options?: PromptOptions) => Promise<string | null>;
  alert: (options?: Pick<ConfirmOptions, "title" | "message">) => Promise<void>;
}

export const ModalContext = createContext<ModalApi | null>(null);

export function useModal(): ModalApi {
  const ctx = useContext(ModalContext);
  if (!ctx) throw new Error("useModal must be used inside <ModalProvider>");
  return ctx;
}
