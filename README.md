# 🐈‍⬛ Kuro-Neko (kuro-neko-llm)

**Kuro-Neko** es un motor de entrenamiento e inferencia de Modelos de Lenguaje (LLM) de arquitectura **Qwen (Qwen2.5 / Qwen3)** escrito totalmente en **Rust**, acelerado por GPU mediante **Burn 0.21** y **WGPU**, con soporte directo para exportación binaria a formato **GGUF v3** (compatible con LM Studio, llama.cpp y Ollama).

---

## 🚀 Características Principales

- **Arquitectura Causal Transformer (Qwen2.5 / Qwen3)**:
  - Atención Causal Multi-Cabeza con **RoPE** ($\theta = 10.000$), sesgo activo en proyecciones QKV (`with_bias(true)`) y máscara triangular `triu_mask`.
  - Capas MLP con **SwiGLU** (activaciones SiLU alineadas a múltiplos de 256).
  - Normalización **RMSNorm** con precision $\epsilon = 10^{-6}$.
  - **Tied Word Embeddings** con inicialización gaussiana normal $\mathcal{N}(0, 0.02)$ para evitar saturación de logits.
  - Regularización **Dropout** y **Weight Decay** ($0.1$).
- **Aceleración GPU con WGPU / Burn 0.21**: Ejecución multiplataforma en GPU (Vulkan, Metal, DX12, WebGPU).
- **Tokenización Paralela Streaming**: Carga streaming multihilo línea por línea con tokenizador BPE Byte-Level, estimación dinámica de capacidad y reporte de errores.
- **Pipeline de Entrenamiento Optimizado**:
  - **Shuffling de datos** por época (Fisher-Yates) para mejor convergencia.
  - **Gradient Clipping** por norma $L_2$ (`Norm(1.0)`) y **Weight Decay** (`0.1`).
  - **Schedule de LR Cosine** con Warmup configurable.
  - **Cálculo Esparso de Pérdida ($NLL$)** con `log_softmax` + `gather`.
  - **Acumulación de Pérdida en GPU** sin stalls de sincronización GPU $\to$ CPU por lote.
- **Exportador GGUF v3 Nativo**: Serialización directa de todos los tensores de pesos y biases del modelo alineados a 32 bytes con cabeceras GGUF v3 y clave `general.alignment`.
- **CLI Moderno con Clap**: Interfaz CLI basada en derive macros con validación estricta y `--help` automático.

---

## 🛠️ Requisitos Previos

- **Rust**: 1.80+ en canal **Stable** (`cargo` y `rustc`).
- **Drivers de GPU**: Soporte Vulkan / Metal / DX12.

---

## ⚙️ Uso de la Línea de Comandos (CLI)

### 1. Entrenar el Modelo (`train`)

Ejecuta el entrenamiento acelerado por GPU usando un dataset de Hugging Face o un archivo local:

```bash
# Entrenar en TinyStories (Configuración por defecto: 10M tokens, batch 128, 3 épocas)
cargo run --release -- train --dataset roneneldan/TinyStories
```

#### Opciones del Entrenador (CLI Flags)

| Flag | Descripción | Valor por Defecto |
|:---|:---|:---|
| `--dataset <repo>` | Repositorio(s) de Hugging Face (ej. `roneneldan/TinyStories`) | `[]` |
| `--file <path>` | Archivo de texto plano local o binario (`.bin`) para entrenar | `None` |
| `--dataset-file <filename>` | Nombre de archivo dentro del repo de HF | `TinyStories-train.txt` |
| `--preset <name>` | Preset de arquitectura: `nano` (~2M), `tiny` (~5.5M), `small` (~30M), `base` (~100M) | `None` |
| `--dtype <f32\|f16>` | Precisión de cómputo. `f16` acelera en GPUs compatibles (backend sin fusión); requiere LR estables | `f32` |
| `--batch-size <n>` | Tamaño de lote (Batch size) | `128` |
| `--epochs <n>` | Número de épocas de entrenamiento | `3` |
| `--max-tokens <n>` | Máximo de tokens a cargar (`0` = sin límite). La referencia Chinchilla es ~20 tokens/parámetro | `100000000` |
| `--step <n>` | Paso de ventana deslizante (`None` = `n-ctx`) | `None` |
| `--n-ctx <n>` | Ventana de contexto del modelo | `256` |
| `--n-embd <n>` | Dimensión de embedding | `256` |
| `--n-head <n>` | Cabezas de atención causal (Query) | `4` |
| `--n-kv-head <n>` | Cabezas Key/Value para GQA (`None` = igual a `n-head`) | `None` |
| `--n-layer <n>` | Número de bloques Transformer | `4` |
| `--lr <float>` | Tasa de aprendizaje (Learning rate) | `0.0003` |
| `--dropout <float>` | Probabilidad de dropout | `0.1` |
| `--attn-softcap <float>` | Capping de atención (`0.0` desactivado, `50.0` Gemma 4) | `0.0` |
| `--val-split <float>` | Ratio reservado para validación y cálculo de perplejidad ($PPL$) | `0.1` |
| `--grad-accum <n>` | Micro-batches acumulados por paso de AdamW (media real de gradientes, equivalente a un batch mayor) | `1` |
| `--save-dir <path>` | Directorio para guardado automático de checkpoints por época | `None` |
| `--resume <path>` | Checkpoint desde el que reanudar (restaura pesos, AdamW, schedule y RNG; `--epochs` = épocas adicionales) | `None` |
| `--seq-warmup-steps <n>` | Pasos iniciales para **Progressive Sequence Scaling** (de $N/4$ a $N$) | `0` |
| `--early-stop-patience <n>` | Detiene el entrenamiento tras `n` épocas sin mejorar `val_loss` (`0` = desactivado) | `0` |

#### Ejemplos de Entrenamientos Personalizados

```bash
# Entrenamiento rápido de prueba con 1M tokens y GQA (n_kv_head=2)
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 1000000 --n-kv-head 2

# Entrenar sobre TODO el dataset sin límite de tokens
cargo run --release -- train --dataset roneneldan/TinyStories --max-tokens 0

# Configurar entrenamiento con Sequence Length Warmup (Progressive Scaling), Acumulación y Checkpointing
cargo run --release -- train --file corpus.txt --seq-warmup-steps 1000 --grad-accum 4 --val-split 0.1 --save-dir checkpoints

# Configurar un modelo más grande con GQA y Softcapping
cargo run --release -- train --dataset roneneldan/TinyStories --n-embd 512 --n-head 8 --n-kv-head 2 --n-layer 8 --attn-softcap 50.0
```

---

### 2. Pre-tokenizar Dataset Binario (`prepare-dataset`)

Pre-tokeniza un corpus o dataset de Hugging Face a un archivo binario `.bin` ultracompacto para carga instantánea con `mmap`:

```bash
cargo run --release -- prepare-dataset --file corpus.txt --output datasets/corpus.bin
```

> [!NOTE]
> Los datasets `.bin` v2 incluyen un *fingerprint* del vocabulario del tokenizador. Si reentrenas el tokenizador, vuelve a ejecutar `prepare-dataset`: cargar un `.bin` tokenizado con otro vocabulario ahora falla con un error explícito en lugar de entrenar con IDs corruptos.

---

### 2b. Reanudar Entrenamiento y Parada Segura

El progreso del entrenamiento se persiste en `train_state.json` (posición del schedule, RNG del shuffler, mejor `val_loss`) y `optimizer_state.bin` (momentos de AdamW), junto a los pesos:

```bash
# Reanudar 2 épocas adicionales desde el checkpoint de la época 3
cargo run --release -- train --dataset roneneldan/TinyStories --resume checkpoints/checkpoint_epoch_3.bin --epochs 2

# Detener con Ctrl+C o desde la web guarda checkpoint_interrupted.bin + estado completo
```

Con `--resume`, `--epochs` significa **épocas adicionales** y el schedule coseno continúa donde se quedó. Si el checkpoint es antiguo (sin sidecars), se cargan solo los pesos con un aviso.

---

### 3. Evaluar Modelo, Perplejidad y BPB (`eval`)

Evalúa la pérdida de entropía cruzada, la Perplejidad ($PPL = e^{\text{loss}}$) y los **bits por byte** (BPB, comparable entre tokenizers) de un modelo entrenado. Por defecto evalúa **todos** los batches del dataset (`--eval-batches 0`); el BPB no está disponible con `.bin` porque no conservan el tamaño original en bytes:

```bash
cargo run --release -- eval --model-dir model_output --file datasets/corpus.txt
```

---

### 4. Entrenar Tokenizador Custom (`train-tokenizer`)

Entrena un tokenizador BPE Byte-Level desde cero sobre un corpus específico:

```bash
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000
```

#### Opciones del Tokenizador (CLI Flags)

| Flag | Descripción | Valor por Defecto |
|:---|:---|:---|
| `--dataset <repo>` | Repositorio(s) de Hugging Face (comma-separated, ej. `roneneldan/TinyStories`) | `[]` |
| `--file <path>` | Archivo de texto plano local para entrenar el vocabulario | `None` |
| `--dataset-file <filename>` | Nombre del archivo objetivo dentro del repositorio del dataset de HF | `TinyStories-train.txt` |
| `--vocab-size <n>` | Tamaño total de la tabla de vocabulario resultante | `8000` |

---

### 5. Generar Texto / Inferencia (`generate`)

Genera texto autorregresivo a partir de una frase semilla usando el modelo entrenado, con muestreo Top-K, Top-P, Min-P y Repetition Penalty:

```bash
cargo run --release -- generate "Once upon a time" --temp 0.9 --top-k 60 --top-p 0.9 --min-p 0.1 --max-tokens 50
```

#### Opciones del Generador (CLI Flags)

| Flag / Argumento | Descripción | Valor por Defecto |
|:---|:---|:---|
| `<prompt>` | Frase semilla (prompt) para iniciar la generación (Posicional) | *Requerido* |
| `--temp <float>` | Temperatura de muestreo para controlar la creatividad/aleatoriedad | `0.9` |
| `--top-k <n>` | Límite superior de tokens candidatos a evaluar (Muestreo Top-K) | `60` |
| `--top-p <float>` | Umbral acumulado de probabilidad para muestreo Nucleus (Top-P) | `0.9` |
| `--min-p <float>` | Umbral de probabilidad relativa Min-P | `0.1` |
| `--rep-penalty <float>` | Factor de penalización por repetición de tokens | `1.6` |
| `--max-tokens <n>` | Cantidad máxima de nuevos tokens a generar | `50` |
| `--seed <n>` | Semilla del generador pseudoaleatorio para reproducibilidad | `None` |

---

### 6. Inspeccionar Estadísticas del Modelo (`stats`)

Muestra el número total de parámetros, huella de memoria FP32, FLOPs por token y compresión GQA:

```bash
cargo run --release -- stats
```

---

### 7. Exportar a Formato GGUF (`export`)

Exporta los pesos del modelo guardados en `model_output` directamente a un binario **GGUF v3**:

```bash
cargo run --release -- export --model-name kuro-neko-2026
```

El archivo `.gguf` resultante se guardará en `exported/kuro-neko-2026.gguf` y se puede abrir directamente en **LM Studio**, **llama.cpp** u **Ollama**.

#### Opciones de Exportación (CLI Flags)

| Flag | Descripción | Valor por Defecto |
|:---|:---|:---|
| `--model-name <name>` | Nombre base del archivo `.gguf` resultante | `kuro-neko-qwen` |

---

## 🏗️ Arquitectura del Proyecto

```
kuro-neko/
├── Cargo.toml          # Dependencias (Burn 0.21, tokenizers, hf-hub, serde, clap)
├── src/
│   ├── main.rs         # Punto de entrada y CLI con Clap derive
│   ├── config.rs       # Estructura ModelConfig unificada
│   ├── model.rs        # Arquitectura LLaMA (Llm, Block, Attention, SwiGLU, Weight Tying)
│   ├── data.rs         # LlmTokenizer y carga streaming multihilo
│   ├── train.rs        # Bucle GPU con Shuffling, Grad Clipping, Weight Decay y AdamW
│   ├── generate.rs     # Inferencia autorregresiva con bounds check y XorshiftRng*
│   ├── export.rs       # Serializador binario GGUF v3 (pesos alineados a 32 bytes)
│   └── tui.rs          # Utilidades visuales de terminal (ANSI, cajas Unicode y progreso)
```
