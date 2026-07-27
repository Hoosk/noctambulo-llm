# 📖 Kuro-Neko — Documentación

Bienvenido a la documentación oficial de **Kuro-Neko** (`kuro-neko-llm`), un motor de entrenamiento e inferencia de Modelos de Lenguaje de Gran Escala (LLM) de arquitectura **Qwen (Qwen2.5 / Qwen3)** desarrollado íntegramente en **Rust** y acelerado por GPU mediante **Burn 0.21** y **WGPU**.

Este conjunto de documentos proporciona una guía detallada sobre la arquitectura del sistema, el uso de la interfaz de línea de comandos (CLI), las estrategias de muestreo de texto, el entrenamiento de tokenizadores y la exportación de modelos al formato estándar **GGUF v3**.

---

## 📚 Índice de Documentación

| Documento | Descripción |
| :--- | :--- |
| 🏋️ **[Entrenamiento (train.md)](train.md)** | Manual completo del subcomando `train`. Explicación detallada de los 15 parámetros de configuración, optimizador AdamW, schedule de tasa de aprendizaje coseno con warmup, gradient clipping y ejemplos prácticos. |
| 🔤 **[Generación e Inferencia (generate.md)](generate.md)** | Guía de uso del subcomando `generate`. Explicación de estrategias de muestreo (Temperature, Top-K, Top-P, Min-P, Repetition Penalty) y funcionamiento de la Caché KV (Prefill & Decode). |
| 🧩 **[Tokenizador Custom (train-tokenizer.md)](train-tokenizer.md)** | Guía de entrenamiento del tokenizador BPE a nivel de bytes mediante el subcomando `train-tokenizer`, normalización Unicode NFKC y manejo de tokens especiales. |
| 📦 **[Exportación GGUF v3 (export.md)](export.md)** | Manual de conversión del modelo al formato estándar `GGUF v3` mediante el subcomando `export`, alineación de tensores a 32 bytes e integración con Ollama, LM Studio y llama.cpp. |
| 📊 **[Estadísticas del Modelo (stats.md)](stats.md)** | Manual del subcomando `stats` para calcular el número total de parámetros, huella de memoria FP32, estimación de FLOPs por token y ahorro de memoria GQA. |
| 🧠 **[Arquitectura Técnica (architecture.md)](architecture.md)** | Especificación técnica profunda de la arquitectura Qwen Transformer: RoPE ($\theta=10.000$), SwiGLU alineado a 256, GQA, RMSNorm ($\epsilon=10^{-6}$), Tied Embeddings y Attention Softcapping. |

---

## ⚡ Flujo de Trabajo Rápido (Quickstart)

El ciclo de vida completo del desarrollo de un modelo en Kuro-Neko consta de los siguientes 5 pasos:

```bash
# 1. Entrenar un tokenizador BPE custom de 8,000 tokens
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000

# 2. Entrenar el modelo causal Transformer con aceleración GPU (WGPU)
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 10000000 --batch-size 128 --epochs 3

# 3. Generar texto autorregresivo desde una semilla
cargo run --release -- generate "Once upon a time" --temp 0.7 --top-k 40 --top-p 0.9 --min-p 0.05

# 4. Inspeccionar la arquitectura y consumo de memoria del modelo
cargo run --release -- stats

# 5. Exportar el modelo entrenado a formato GGUF v3 para inferencia externa
cargo run --release -- export --model-name kuro-neko-v1
```

---

## 🔗 Referencias Cruzadas

- Para consultar el manual principal del repositorio, revisa el [README.md raíz](../README.md).
- Si deseas entender la estructura del código fuente, revisa el apartado de módulos en [architecture.md](architecture.md).
