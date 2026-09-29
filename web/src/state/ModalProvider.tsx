import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Dialog, type DialogClose } from "../components/Dialog";
import { ModalContext, type DialogState, type ModalApi } from "./modalContext";

export function ModalProvider({ children }: { children: ReactNode }) {
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const dialogRef = useRef<DialogState | null>(null);

  useEffect(() => {
    dialogRef.current = dialog;
  }, [dialog]);

  const close = useCallback<DialogClose>((result) => {
    const current = dialogRef.current;
    if (!current) return;
    dialogRef.current = null;
    setDialog(null);
    if (current.kind === "confirm") current.resolve(result === true);
    else if (current.kind === "prompt") current.resolve(typeof result === "string" ? result : null);
    else current.resolve();
  }, []);

  const api = useMemo<ModalApi>(
    () => ({
      confirm: (options = {}) =>
        new Promise<boolean>((resolve) => {
          setDialog({ kind: "confirm", options, resolve });
        }),
      prompt: (options = {}) =>
        new Promise<string | null>((resolve) => {
          setDialog({ kind: "prompt", options, resolve });
        }),
      alert: (options = {}) =>
        new Promise<void>((resolve) => {
          setDialog({ kind: "alert", options, resolve });
        }),
    }),
    [],
  );

  return (
    <ModalContext.Provider value={api}>
      {children}
      {dialog ? <Dialog dialog={dialog} onClose={close} /> : null}
    </ModalContext.Provider>
  );
}
