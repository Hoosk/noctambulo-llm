# 🗺️ Kuro-Neko Roadmap

Last updated: 2026-10-03 (KV cache modernization started: Phase 0 quality gates and Phase 1 preallocated cache/f16 generation landed).

## Current Status

- Complete engine: BPE tokenizer, datasets (txt/bin/parquet with direct ingest and dedupe), `train` (real gradient accumulation, full-state resume, safe stop, f16, RoPE context extension, sliding window + sinks, cross-layer KV sharing, MLA latent cache, label smoothing, z-loss, AdamW weight decay), `eval` (BPB, all batches, early stopping, `--max-tokens` slice), `generate` (CLI + SSE streaming, RoPE scaling flags, f32/f16, bounded ring cache), GGUF `export` (with `rope.scaling.*` and `attention.sliding_window` keys, `--approx` for grouped models, exact folding for MLA) and the `serve` web suite.
- 81 tests green (`cargo test`), `cargo clippy --all-targets` clean, ESLint + react-doctor 100/100 in `web/`, and CDP browser smoke tests on the built bundle.
- Fixed correctness bugs found while testing the f16 cache:
  1. **M=1 GEMV**: burn-wgpu's autotuned cubek GEMV kernel returns wrong results for output widths >= 32, so token-by-token decode silently diverged from the batched forward. Autotune must stay enabled (without it the fallback f16 kernels collapse training), so the model now **pads single-token forwards to M=2** before every matmul and slices row 0 back out (`Llm::forward`/`forward_with_cache`). Regression test `test_wgpu_cached_decode_matches_full_forward` covers prefill + decode on both backends.
  2. **f16 tensor creation**: Burn 0.21's `from_data`/`zeros`/`ones` create f32 tensors even on an f16 backend, so f16 kernels read f32 buffers. `Llm::align_compute_dtype` (applied after init and every `load_record`), the RoPE tables and the cache scales now cast to the backend dtype; `Tensor::dtype()` no longer lies.
  3. **f16 cache storage**: the fused `Wgpu` backend's f16 casts are silent no-ops (dtype reports f16, values stay f32). `generate --kv-cache-dtype f16` is routed to the non-fused `WgpuHalf` backend, and `KvCache::preallocated` probes the backend and panics instead of silently wasting the memory saving.
  4. **f16 training**: Burn 0.21's AdamW moment estimates underflow in f16 (the first update explodes and the loss stays at ln(vocab)); `train --dtype f16` is rejected with a clear error. Training is f32-only.
- Quantized cache (`--kv-cache-dtype int8|q4`): packed i32 words (4 int8 or 8 int4) plus per-block f32 scales, written with `slice_assign` and dequantized on read. Burn 0.21's `QFloat` lacks `slice`/`slice_assign`, so the packed representation sidesteps it and works on every backend. Tested for parity against the f32 cache (standard + MLA + window) and for real storage; int8 decode matches f32 bit-for-bit on the trained models.
- Web: React 19 + Vite + TypeScript + Tailwind v4 built locally (no CDN), English UI, HttpOnly-cookie auth, served from `web/dist` by Axum. Train/Datasets/Generate/Models/Export/Devices views plus run history and live telemetry.
- Docs: English-only (README, `cli_parameters.md`, `doc/*.md`, ROADMAP and the auxiliary notes).

## KV Cache Modernization (DeepSeek-style)

Goal: move from a per-layer f32 cache that reallocated with `Tensor::cat` on every decoded token to a compressed, bounded, optionally quantized cache. Plan and status:

- ✅ **Phase 0 — Baseline and quality gates**: cache-vs-full-forward parity test, overfit regression test (caught the Stage D weight-decay bug: a `ModuleMapper`-based decay froze every mapped parameter), `stats` KV bytes/token, `eval --max-tokens`, decode benchmark.
- ✅ **Phase 1 — Cache engine v2**: preallocated per-layer buffers written with in-place `slice_assign` (no more per-token `cat`), attention over the filled region only, `generate --dtype f16` (f16 cache follows compute), API/web compute-precision field. Decode baseline on the tiny model: ~8.6 tok/s, ~143 MB peak RSS.
- ✅ **Phase 2 — Sliding window + sinks + dynamic RoPE**: `sliding_window`/`sink_tokens` in `ModelConfig` (CLI/API/web), window+sink masks in the training and cached forwards, RoPE angles computed per position (the precomputed table and its 16k cap are gone) and a bounded ring-buffer KV cache written in place with chunked prefill. Verified end-to-end: a tiny 1-epoch model with `--sliding-window 64 --sink-tokens 4` reaches PPL 373 (random is ~8000), generates coherent TinyStories text, and continues past the window with a 68-row cache. GGUF exports `llama.attention.sliding_window`.
- ✅ **Phase 3 — Cross-layer KV sharing**: one `KvProj` and one cache per group (`kv_share_group`, CLI/API/web), attention split into query/attend so group leaders project and write K/V once while followers reuse it. `arch_version` (now 3) rejects incompatible checkpoints with a clear error, `param_count`/`stats` account for groups, and GGUF export refuses grouped models unless `--approx` is passed (documented standard-attention approximation). Verified: 4-layer/group-2 tiny model reaches PPL 329 and coherent generation; export gate and `--approx` output both checked.
- ✅ **Phase 4 — MLA latent cache**: `kv_down`/`k_up`/`v_up` per group, latent-only cache (the `v` buffer is not even allocated), K/V expanded on the fly from the cached latent, `--mla-latent-dim` in CLI/API/web and `param_count`/`stats` updated. Simplification vs DeepSeek: no decoupled RoPE and no matrix absorption yet, which makes the folded export (`kv_down · k_up`) an **exact** standard-attention equivalent (no `--approx` needed). Verified: 4-layer latent-32 model reaches PPL 322, coherent generation, cache 512 B/token vs 2048 B/token standard (4×), exact GGUF export (43 tensors).
- ✅ **Phase 5 — Quantized KV cache**: `generate --kv-cache-dtype f32|f16|int8|q4`. f16 halves memory on the non-fused backend (the fused backend's casts are no-ops; it is detected and rejected). int8/q4 store packed i32 words (4 int8 or 8 int4) plus per-block f32 scales and dequantize on read; Burn 0.21's `QFloat` cannot be written with `slice`/`slice_assign`, so the packed representation sidesteps it and stays portable across backends. `stats` reports bytes/token for every format. Custom `burn::cubecl` kernels remain a possible optimization for the pack/unpack elementwise work, but are not required for correctness or the memory saving.
- ✅ **GGUF import & model recovery**: `import` (CLI) and `POST /api/models/import` invert the exporter exactly and rebuild the tokenizer from the GGUF vocab, so models that only survive as GGUF are usable again. Generation/eval/export resolve `custom_tokenizer.json` from the model directory, and `/api/models` reports architecture compatibility, training steps and best validation loss.
- ✅ **Train wizard & system info**: the Train form is a four-step wizard (data → architecture preset → schedule → review) with inline validation and a live parameter count; f16 is no longer offered for training. The Devices view adds a host system card (CPU model, cores, frequency, load, OS/kernel, RAM, swap, disks) backed by `sysinfo` via `GET /api/system`.
- ⏳ **Phase 6 (backlog)** — heavy-hitter eviction and host-RAM spill/replay.

Known prerequisites discovered in Phase 0: all existing checkpoints are invalid (the v1 model was a 1-step run; v2 was trained with an older tokenizer and served with the current one; every `.bin` dataset is v1 without a fingerprint). Re-run `prepare-dataset` with the current tokenizer and retrain from scratch before measuring quality.

## Stage D — Training Quality and Speed

1. ✅ **Token budget**: default 10M → 100M, Chinchilla-style warning (~20 tokens/parameter) and `nano`/`tiny`/`small`/`base` presets.
2. ✅ **FP16**: `--dtype f32|f16` flag (non-fused backend for f16). `bf16` dropped: WGSL does not support it and cubecl crashes; an explicit error is returned instead. Records are always f32, so generate/export are unaffected.
3. ✅ **Eval**: removed the fixed 100-batch cap (`--eval-batches`, 0 = all), added the BPB (bits-per-byte) metric and early stopping (`--early-stop-patience`).
4. ✅ **RoPE scaling**: `linear`, `ntk` (dynamic NTK) and `yarn` strategies with a configurable factor. The per-layer RoPE cache extends to `factor × n_ctx` (capped), `generate`/`eval` take `--rope-scaling`/`--rope-factor`, the web Generate view exposes both, and GGUF exports carry `llama.rope.scaling.{type,factor,original_context_length}`.
5. ✅ **Datasets**: `prepare-dataset` tokenizes Parquet text columns straight to `.bin` (no intermediate `.txt`) and `--dedupe` drops exact duplicate lines/documents (hashed) for both text and Parquet; mixing Parquet sources is rejected with an explicit error.
6. ✅ **Optimizer**: decoupled weight decay (AdamW) applied to all parameters with `--weight-decay`, plus optional `--label-smoothing` and `--z-loss` (also in the web Train form and API).
7. ✅ **Optional CUDA backend**: `cargo build --features cuda` swaps the CLI (train/eval/generate/export) to `burn::backend::Cuda` for f32 and f16; the web server keeps WGPU for adapter enumeration. Compile-checked with `cargo check --features cuda` (runtime requires an NVIDIA GPU).

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
- ✅ **UI smoke test**: versioned at `web/smoke/run.mjs` (CDP over headless Chromium, no test framework) with `npm run smoke`, configurable URL/token/CDP flags, self-created and self-cleaned artifacts; covers login, D4/D6 form fields, telemetry, model CRUD, GGUF export, weighted combine (`.bin` rejection), run history, worker logs and val curve.
- ✅ **README**: rewritten in English with install, full workflow, web build/auth and troubleshooting.

## Technical Debt (Docs and Strings)

- ✅ **Translate all documentation to English**: README, `cli_parameters.md`, `doc/*.md`, ROADMAP and the auxiliary notes, with no Spanish copies.
- ✅ **Code strings and help text**: the remaining Spanish comments/messages in `src/` were translated; `serve --token` now documents that omitting it disables auth and `train --dtype` states that `bf16` is rejected by the WGPU backend.

## Suggested Order

The KV cache modernization phases above are the active work, in order (5). Separate known gap: the optional CUDA backend is compile-checked (`cargo check --features cuda`) but has not run on an NVIDIA host; validate it there before advertising runtime support. Re-prepare the datasets with the current tokenizer and retrain from scratch (see prerequisites above).
