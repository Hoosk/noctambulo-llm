import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Button } from "./Button";
import { Icon } from "./Icon";
import { Input } from "./ui";
import type { ConfirmOptions, DialogState, PromptOptions } from "../state/modalContext";

export type DialogClose = (result?: boolean | string | null) => void;

function ConfirmBody({ options, onClose }: { options: ConfirmOptions; onClose: DialogClose }) {
  const danger = Boolean(options.danger);
  return (
    <>
      <div className="flex items-start gap-3 mb-4">
        <div
          className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
            danger ? "bg-rose-500/15 text-rose-300" : "bg-violet-500/15 text-violet-300"
          }`}
        >
          <Icon name={danger ? "trash" : "info"} className="w-4 h-4" />
        </div>
        <div className="min-w-0">
          <h4 className="text-sm font-semibold text-white">{options.title ?? "Confirm?"}</h4>
          {options.message ? (
            <p className="text-sm text-slate-400 mt-1 whitespace-pre-wrap break-words">{options.message}</p>
          ) : null}
        </div>
      </div>
      <div className="flex gap-2 justify-end">
        <Button onClick={() => onClose(null)}>Cancel</Button>
        <Button variant={danger ? "danger" : "primary"} onClick={() => onClose(true)} autoFocus>
          {options.confirmText ?? "Confirm"}
        </Button>
      </div>
    </>
  );
}

function AlertBody({ options, onClose }: { options: ConfirmOptions; onClose: DialogClose }) {
  return (
    <>
      <div className="flex items-start gap-3 mb-4">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 bg-violet-500/15 text-violet-300">
          <Icon name="info" className="w-4 h-4" />
        </div>
        <div className="min-w-0">
          <h4 className="text-sm font-semibold text-white">{options.title ?? "Information"}</h4>
          {options.message ? (
            <p className="text-sm text-slate-400 mt-1 whitespace-pre-wrap break-words">{options.message}</p>
          ) : null}
        </div>
      </div>
      <div className="flex justify-end">
        <Button variant="primary" onClick={() => onClose(null)} autoFocus>
          Got it
        </Button>
      </div>
    </>
  );
}

function PromptBody({ options, onClose }: { options: PromptOptions; onClose: DialogClose }) {
  const [value, setValue] = useState(options.value ?? "");
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onClose(value.trim() || null);
  };
  return (
    <form onSubmit={submit}>
      <h4 className="text-sm font-semibold text-white mb-3">{options.title ?? "Input"}</h4>
      {options.label ? (
        <label htmlFor="prompt-input" className="block text-xs text-slate-400 mb-1.5">
          {options.label}
        </label>
      ) : null}
      <Input
        id="prompt-input"
        autoFocus
        value={value}
        placeholder={options.placeholder ?? ""}
        onChange={(e) => setValue(e.target.value)}
        onFocus={(e) => e.currentTarget.select()}
      />
      <div className="flex gap-2 justify-end mt-4">
        <Button onClick={() => onClose(null)}>Cancel</Button>
        <Button type="submit" variant="primary">
          Save
        </Button>
      </div>
    </form>
  );
}

export function Dialog({ dialog, onClose }: { dialog: DialogState; onClose: DialogClose }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 animate-fade-in">
      <button
        type="button"
        aria-label="Close dialog"
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm cursor-default"
        onClick={() => onClose(null)}
      />
      <div className="relative glass-card rounded-2xl p-5 w-full max-w-sm animate-scale-in">
        {dialog.kind === "prompt" ? (
          <PromptBody options={dialog.options} onClose={onClose} />
        ) : dialog.kind === "alert" ? (
          <AlertBody options={dialog.options} onClose={onClose} />
        ) : (
          <ConfirmBody options={dialog.options} onClose={onClose} />
        )}
      </div>
    </div>
  );
}
