# Web smoke tests

End-to-end browser test for the built web console. It drives a headless Chromium over the
Chrome DevTools Protocol (no test framework or browser driver installed) and asserts the
Stage D/E behaviours: login, telemetry, model files/rename/delete, GGUF export with
progress, weighted dataset combine (including `.bin` rejection), RoPE/loss form fields,
training run history, worker logs and the val-loss curve.

## Prerequisites

1. A built frontend and a running server:

   ```bash
   (cd web && npm install && npm run build)
   cargo run --release -- serve --token noctambulo-demo --datasets-dir datasets --models-dir .
   ```

2. A headless Chromium exposing CDP:

   ```bash
   chromium --headless=new --no-sandbox --remote-debugging-port=9222 \
     --user-data-dir=/tmp/noctambulo-smoke-profile about:blank
   ```

3. Node 22+ (global `fetch` and `WebSocket`).

## Run

```bash
npm run smoke                        # from web/
node web/smoke/run.mjs --help        # flags (--url, --token, --cdp, --models-dir, ...)
```

The suite creates and removes its own artifacts: a disposable copy of the model it just
trained (or of `--fixture` when given), a 1 MB dataset mix and a tiny training run. Use
`--keep-artifacts` to inspect them, or `--skip-training` for a fast UI-only pass (with
`--fixture <dir>` so the model CRUD checks have a checkpoint to copy).
