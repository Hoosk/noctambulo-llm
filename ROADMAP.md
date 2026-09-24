# 🗺️ Roadmap de Kuro-Neko

Última actualización: 2026-09-24 (tras `53f6a85` entrenamiento + `ebdf5ce` web, más el pulido de frontend posterior).

## Estado actual

- Motor completo: tokenizer BPE, datasets (txt/bin/parquet), `train` (acumulación real, resume con estado completo, parada segura), `eval`, `generate`, `export` GGUF y `serve` web.
- 31 tests verdes (`cargo test`), ESLint limpio en el frontend, smoke tests de UI con Chromium headless (consola, IDs duplicados, SVG-como-texto).
- Web: SPA vanilla sin build step, con Tailwind y Chart.js por CDN.

## Etapa D — Calidad y velocidad de entrenamiento

1. **Presupuesto de tokens**: subir el default (10M → 100M) y añadir presets de arquitectura.
2. **BF16/FP16**: flag `--dtype` con `Wgpu<f16, i32>`; validar export GGUF (castear a f32).
3. **Eval**: quitar el cap de 100 batches, métrica BPB (bits-per-byte) y early stopping opcional.
4. **RoPE scaling** (linear/NTK/YaRN) para inferencia más allá de `n_ctx` + claves GGUF `rope.scaling.*`.
5. **Datasets**: ingesta Parquet/HF directa a `.bin` en `prepare-dataset` y dedupe opcional.
6. **Optimizador**: param groups de weight decay (excluir norms/biases), z-loss/label smoothing opcionales.
7. **Backend CUDA** opcional como feature.

## Etapa E — Web suite

1. **Generación en streaming** por SSE + botón de cancelar.
2. **CRUD de modelos/checkpoints**: borrar, renombrar, descargar pesos/config y tokenizer por modelo.
3. **Progreso real** de operaciones largas (descarga HF, combine, export).
4. **Telemetría GPU/VRAM** en vivo.
5. **Historial de runs** con curvas comparables.
6. **Datasets**: que `combine` rechace `.bin` (hoy el mezclador es línea-a-línea y produce un mix vacío en silencio); mejorar la edición de pesos.
7. **Pulido**: logs por SSE (eliminar el polling de 1 s), gráfica de val loss/PPL, Tailwind local (sin CDN).

## Deuda técnica (frontend)

- **Framework**: hoy es vanilla + mini-framework casero (`el()` en `core/dom.js`, bus en `core/events.js`, store en `core/store.js`, router en `main.js`, primitivas en `ui.js`). Opciones a decidir:
  - **A. Vanilla + convenciones/lint/tests**: coste bajo, mantiene cero build.
  - **B. Preact + htm por CDN**: reactividad sin build step.
  - **C. React 19 + Vite + Tailwind local**: componentización real y build step; el server serviría `dist/`. Implica migrar ~2.8k LOC y añadir toolchain de Node.
- **Idioma**: la UI está en castellano; se quiere en inglés. Plan: extraer strings a `core/i18n.js` con diccionarios `en`/`es` y `t("clave")`, o reescribir directamente a inglés. Decidir antes de tocar más vistas.
- **Linting**: ESLint se ejecuta hoy con `npx` desde fuera del repo; añadir config + script (`npm run lint`) y/o CI.
- **Smoke test UI**: versionar el script CDP (Chromium headless) que valida errores de consola, IDs duplicados, SVG-como-texto y genera capturas por tab.
- **Convención DOM**: documentar `el()` vs `html:` (los SVG inline siempre por `html:`) para no repetir el bug.
- **README**: reescribir en condiciones (instalación, flujo completo, capturas, troubleshooting, arquitectura).

## Orden sugerido

Commit pendiente → **D1+D3** (más tokens y eval decente) → **E1** (streaming) → decisión de framework/idioma → resto.
