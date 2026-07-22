# 🐈‍⬛ Kuro-Neko (my-rust-llm)

**Kuro-Neko** es un motor de entrenamiento e inferencia de Modelos de Lenguaje (LLM) de arquitectura GPT-2 escrito totalmente en **Rust**, acelerado por GPU mediante **Burn 0.21** y **WGPU**, con soporte directo para exportación binaria a formato **GGUF v3** (compatible con LM Studio, llama.cpp y Ollama).

---

## 🚀 Caracteristicas Principales

- **Arquitectura Causal Transformer (GPT-2)**: Atencion Causal Multi-Cabeza, Capas MLP con activaciones GELU, LayerNorm pre-LN, Embeddings Posicionales y Token Embeddings.
- **Aceleracion GPU con WGPU / Burn 0.21**: Ejecucion multiplataforma en GPU (Vulkan, Metal, DX12, WebGPU).
- **Tokenizacion Paralela Streaming**: Carga streaming multihilo linea por linea con tokenizador BPE Byte-Level y barra de progreso dinamica con ETA.
- **Pipeline de Entrenamiento Ultraoptimizado**:
  - **Perdida Esparsa ($NLL$)**: Calculo de entropia cruzada con `log_softmax` + `gather` con consumo de memoria de 32 KB por lote en lugar de matrices one-hot de 262 MB.
  - **Acumulacion de Perdida en GPU**: Sin stalls de sincronizacion GPU $\to$ CPU por lote.
  - **Mascara Causal Directa**: Asignacion de mascara triangular superior sin pases booleanos adicionales.
  - **Reporte en Tiempo Real**: Estadisticas dinamicas de batches/s, ETA y previsualizacion automatica de historias al finalizar cada epoca.
- **Exportador GGUF v3 Nativo**: Serializacion directa de todos los tensores de pesos del modelo alineados a 32 bytes con cabeceras GGUF v3.
- **Parametrizacion Completa por CLI**: Control de hyperparametros via flags de linea de comandos.

---

## 🛠️ Requisitos Previos

- **Rust**: 1.80+ (`cargo` y `rustc`).
- **Drivers de GPU**: Soporte Vulkan / Metal / DX12.

---

## ⚙️ Uso de la Linea de Comandos (CLI)

### 1. Entrenar el Modelo (`train`)

Ejecuta el entrenamiento acelerado por GPU usando un dataset de Hugging Face o un archivo local:

```bash
# Entrenar en TinyStories (Configuracion optimizada por defecto: 50M tokens, batch 128, 3 epocas)
cargo run --release train --dataset roneneldan/TinyStories
```

#### Opciones de Entrenador (CLI Flags)

| Flag | Descripcion | Valor por Defecto |
|:---|:---|:---|
| `--dataset <repo>` | Repositorio de Hugging Face (ej. `roneneldan/TinyStories`) | `None` |
| `--file <path>` | Archivo de texto local para entrenar | `corpus.txt` |
| `--batch-size <n>` | Tamaño de lote (Batch size) | `128` |
| `--epochs <n>` | Numero de epocas de entrenamiento | `3` |
| `--max-tokens <n>` | Maximo de tokens a cargar (`0` = sin limite) | `50000000` |
| `--step <n>` | Paso de ventana deslizante (`0` o omitido = `n-ctx`) | `n-ctx` |
| `--n-ctx <n>` | Ventana de contexto del modelo | `64` |
| `--n-embd <n>` | Dimension de embedding | `256` |
| `--n-head <n>` | Cabezas de atencion causal | `4` |
| `--n-layer <n>` | Numero de bloques Transformer | `4` |
| `--lr <float>` | Tasa de aprendizaje (Learning rate) | `0.001` |

#### Ejemplos de Entrenamientos Personalizados

```bash
# Prueba ultra-rapida de 2 minutos (10M tokens)
cargo run --release train --dataset roneneldan/TinyStories --max-tokens 10000000

# Entrenar sobre TODO el dataset sin limite de tokens
cargo run --release train --dataset roneneldan/TinyStories --max-tokens 0

# Configurar un modelo mas grande (512 embd, 8 heads, 8 layers)
cargo run --release train --dataset roneneldan/TinyStories --n-embd 512 --n-head 8 --n-layer 8 --batch-size 64
```

---

### 2. Entrenar Tokenizador Custom (`train-tokenizer`)

Entrena un tokenizador BPE Byte-Level desde cero sobre un corpus especifico:

```bash
cargo run --release train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000
```

---

### 3. Generar Texto / Inferencia (`generate`)

Genera texto autorregresivo a partir de una frase semilla usando el modelo entrenado:

```bash
cargo run --release generate "Once upon a time"
```

---

### 4. Exportar a Formato GGUF (`export`)

Exporta los pesos del modelo guardados en `model_output` directamente a un binario **GGUF v3**:

```bash
cargo run --release export --outfile model_output/model.gguf
```

El archivo `.gguf` resultante se puede abrir directamente en **LM Studio**, **llama.cpp** u **Ollama**.

---

## 🏗️ Arquitectura del Proyecto

```
kuro-neko/
├── Cargo.toml          # Dependencias (Burn 0.21, tokenizers, hf-hub, serde)
├── src/
│   ├── main.rs         # Punto de entrada y parsing de CLI flags
│   ├── config.rs       # Estructura ModelConfig con validaciones defensivas
│   ├── model.rs        # Arquitectura Transformer Causal (Llm, Block, Attention, Mlp)
│   ├── data.rs         # LlmTokenizer y carga streaming multihilo con ETA
│   ├── train.rs        # Bucle de entrenamiento GPU, acumulacion y perdida esparsa
│   ├── generate.rs     # Motor de muestreo autorregresivo (temperature, top-p/greedy)
│   └── export.rs       # Serializador binario nativo GGUF v3 (pesos y metadatos)
```

---

## 📊 Diseno Tecnico y Optimización de Rendimiento

1. **Memoria Zero-Copy**: Uso de `std::mem::take` durante el streaming de tokenizacion para transferir ownership de lotes de strings sin clonar en el Heap.
2. **Optimizacion de Atencion Causal**: Mascara triangular superior directa construida mediante `arange.unsqueeze_dim(0).greater(arange.unsqueeze_dim(1))` que evita inversiones booleanas redundantes en cada capa.
3. **Calculo Esparso de Perdida**: Implementado como `loss = -log_softmax(logits).gather(targets).mean()`, lo que elimina la creacion de matrices `one_hot` gigantes en la memoria VRAM de la GPU.
4. **Sincronizacion GPU Incremental**: Las llamadas a `.into_data()` solo se realizan en intervalos de reporte programados, manteniendo la canalizacion (pipeline) de WGPU al 100% de saturacion sin stalls.
