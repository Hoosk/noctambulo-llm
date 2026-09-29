# 🗺️ Kuro-Neko Roadmap

Last updated: 2026-09-29 (after the Stage E web suite: model CRUD, background-operation progress, telemetry, run history and log streaming).

## Current Status

- Complete engine: BPE tokenizer, datasets (txt/bin/parquet), `train` (real gradient accumulation, full-state resume, safe stop, f16), `eval` (BPB, all batches, early stopping), `generate`, GGUF `export` and the `serve` web suite.
- 53 tests green (`cargo test`), `cargo clippy --all-targets` clean, ESLint + react-doctor 100/100 in `web/`, and CDP browser smoke tests on the built bundle.
- Web: React 19 + Vite + TypeScript + Tailwind v4 built locally (no CDN), English UI, HttpOnly-cookie auth, served from `web/dist` by Axum. Train/Datasets/Generate/Models/Export/Devices views plus run history and live telemetry.
- Docs: English-only (README, `cli_parameters.md`, `doc/*.md`, ROADMAP and the auxiliary notes).

## Stage D — Training Quality and Speed

1. ✅ **Token budget**: default 10M → 100M, Chinchilla-style warning (~20 tokens/parameter) and `nano`/`tiny`/`small`/`base` presets.
2. ✅ **FP16**: `--dtype f32|f16` flag (non-fused backend for f16). `bf16` dropped: WGSL does not support it and cubecl crashes; an explicit error is returned instead. Records are always f32, so generate/export are unaffected.
3. ✅ **Eval**: removed the fixed 100-batch cap (`--eval-batches`, 0 = all), added the BPB (bits-per-byte) metric and early stopping (`--early-stop-patience`).
4. **RoPE scaling** (linear/NTK/YaRN) for inference beyond `n_ctx` + GGUF `rope.scaling.*` keys.
5. **Datasets**: direct Parquet/HF ingestion to `.bin` in `prepare-dataset` and optional dedupe.
6. **Optimizer**: weight-decay parameter groups (exclude norms/biases), optional z-loss/label smoothing.
7. **Optional CUDA backend** as a feature.

## Stage E — Web Suite

1. ✅ **Streaming generation** over SSE (`/api/generate/stream`) with cancellation on disconnect and a "Stop" button in the UI + live tok/s.
2. ✅ **Model CRUD**: list/use/remove plus rename, delete and per-file download (`/api/models/files`, `/api/models/file`, `/api/models/rename`, `DELETE /api/models/:name`); mutations are refused while training runs and are restricted to direct children of the models root.
3. ✅ **Real progress** for long operations: HF downloads (byte-level), weighted combine (chars) and GGUF export (tensors) run as background ops with an SSE event stream (`/api/ops/stream`), a snapshot endpoint and cooperative cancellation.
4. ✅ **Live telemetry**: `/api/telemetry` reports process CPU/RSS/threads/uptime, system memory and GPU memory counters when the OS exposes them (nvidia-smi or DRM sysfs); the Devices view polls it every 2 s and states clearly when VRAM counters are unavailable.
5. ✅ **Run history**: every job is persisted to `.kuro/runs.jsonl` with its config, final loss/PPL and train/val curves; the Train view overlays up to 3 selected runs.
6. ✅ **Datasets**: weighted mixing rejects `.bin` sources with an explicit error (shared by CLI and web); the combine card disables itself, lists the offending files and shows each source's normalized share with an "Equalize weights" shortcut.
7. ✅ **Polish**: worker logs stream over SSE (the 1 s polling is gone, backfill via a monotonic cursor on `/api/train/status`), val-loss curve persisted alongside the training loss and drawn as a dashed series; Tailwind built locally (no CDN).

## Technical Debt (Web)

- ✅ **Framework**: option C chosen and executed — React 19 + Vite + Tailwind v4, with the server serving `dist/`; the old vanilla `frontend/` (~2.8k LOC) was deleted.
- ✅ **Language**: UI rewritten in English; docs are English-only.
- ✅ **Linting**: ESLint flat config with `npm run lint` plus `npm run doctor` (react-doctor, kept at 100/100).
- **UI smoke test**: version the CDP (headless Chromium) script in the repo; today it lives in scratch space and validates console errors and each tab.
- ✅ **README**: rewritten in English with install, full workflow, web build/auth and troubleshooting.

## Technical Debt (Docs and Strings)

- ✅ **Translate all documentation to English**: README, `cli_parameters.md`, `doc/*.md`, ROADMAP and the auxiliary notes, with no Spanish copies.
- **Code strings and help text**: parts of `src/` still mix Spanish comments and occasional messages; migrate them to English. Also fix inaccurate `--help` strings (`serve --token` claims it is auto-generated; `train --dtype` advertises `bf16` although runtime rejects it).

## Suggested Order

Remaining work is Stage D: **D4** (RoPE scaling) → **D5** (Parquet/HF → `.bin` + dedupe) → **D6** (optimizer groups, z-loss) → **D7** (optional CUDA). In parallel: version the UI smoke test in the repo and finish the English pass over code strings and `--help` text.
