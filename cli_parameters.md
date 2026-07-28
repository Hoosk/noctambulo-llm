# 🐈‍⬛ Kuro-Neko — Diccionario de Parámetros de la Línea de Comandos (CLI)

Este archivo contiene el diccionario de todos los parámetros aceptados por los diferentes subcomandos de la CLI de **Kuro-Neko**.

---

## 🏋️ 1. Parámetros del Subcomando `train`
Ejecuta el pipeline de entrenamiento de la red neuronal en la GPU usando WGPU.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | ID de dataset(s) en Hugging Face Hub (separados por comas, ej. `roneneldan/TinyStories`). |
| `--file` | `Option<String>`| `None` | Ruta a un archivo de texto plano local o binario (`.bin`) para entrenar. |
| `--dataset-file` | `String` | `"TinyStories-train.txt"` | Nombre del archivo objetivo dentro del repositorio del dataset de HF. |
| `--batch-size` | `usize` | `128` | Tamaño del lote (batch size) a procesar en la GPU antes de la actualización de pesos. |
| `--epochs` | `usize` | `3` | Número de pasadas completas a través del conjunto de datos. |
| `--max-tokens` | `usize` | `10000000` | Límite máximo de tokens a cargar desde el corpus. Un valor de `0` entrena con el archivo completo. |
| `--step` | `Option<usize>`| `None` | Desplazamiento de la ventana deslizante (sliding window step). Si es menor que `n_ctx` genera solapamiento (overlap). `None` equivale a `n_ctx`. |
| `--n-ctx` | `usize` | `256` | Longitud máxima de la ventana de contexto del modelo en tokens. |
| `--n-embd` | `usize` | `256` | Dimensión de embedding y estados ocultos. Debe ser divisible entre `n-head`. |
| `--n-head` | `usize` | `4` | Cantidad de cabezas de atención Query ($Q$). |
| `--n-kv-head` | `Option<usize>`| `None` | Cantidad de cabezas Key/Value ($K, V$). Activa GQA si es menor que `n-head`. `None` o `0` activa MHA estándar. |
| `--n-layer` | `usize` | `4` | Número de bloques Transformer apilados. |
| `--lr` | `f64` | `0.0003` | Tasa de aprendizaje máxima (peak learning rate) para AdamW. |
| `--dropout` | `f64` | `0.1` | Probabilidad de desactivación de neuronas para la regularización de la atención y MLP. |
| `--attn-softcap` | `f32` | `0.0` | Umbral para acotamiento suave de puntuaciones de atención mediante $\tanh$. `0.0` lo desactiva. |
| `--val-split` | `f32` | `0.1` | Porcentaje del dataset reservado para validación y cálculo de perplejidad ($\text{PPL} = e^{\text{loss}}$). |
| `--grad-accum` | `usize` | `1` | Pasos de acumulación de gradientes antes de ejecutar el paso de AdamW. |
| `--save-dir` | `Option<String>`| `None` | Directorio para guardar checkpoints automáticos del modelo al final de cada época. |
| `--resume` | `Option<String>`| `None` | Ruta a un archivo de checkpoint o pesos previos para reanudar el entrenamiento. |
| `--seq-warmup-steps` | `usize` | `0` | Pasos iniciales para **Progressive Sequence Scaling** (hace crecer la ventana de contexto desde $N/4$ hasta $N$). `0` lo desactiva. |

---

## ⚡ 2. Parámetros del Subcomando `prepare-dataset`
Pre-tokeniza un corpus de texto plano o dataset de HF a un formato binario ultracompacto `.bin` con cabecera de metadatos `KNKB`.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | ID de dataset(s) de HF a descargar y pre-tokenizar. |
| `--file` | `Option<String>`| `None` | Ruta al archivo de texto local a convertir. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | Nombre del archivo dentro del repo de HF. |
| `--output` | `String` | `"datasets/dataset.bin"` | Ruta del archivo binario de salida resultante (`.bin`). |
| `--max-tokens` | `usize` | `0` | Límite máximo de tokens a guardar (`0` = sin límite). |

---

## 📊 3. Parámetros del Subcomando `eval`
Evalúa la pérdida de entropía cruzada y perplejidad ($PPL$) de un modelo guardado sobre un dataset.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `--model-dir` | `String` | `"model_output"` | Directorio que contiene `config.json` y `model_weights.bin`. |
| `--dataset` | `Vec<String>` | `[]` | ID de dataset(s) de HF para evaluar. |
| `--file` | `Option<String>`| `None` | Ruta al archivo de texto o binario `.bin` de evaluación. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | Nombre del archivo objetivo en HF. |
| `--n-ctx` | `Option<usize>`| `None` | Sobrescribe la ventana de contexto para la evaluación. |

---

## 🔤 4. Parámetros del Subcomando `train-tokenizer`
Entrena un tokenizador BPE a nivel de bytes en base a un corpus de texto.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `--dataset` | `Vec<String>` | `[]` | ID de dataset(s) de HF a descargar para usarse como corpus base. |
| `--file` | `Option<String>`| `None` | Ruta a un archivo de texto local para entrenar el vocabulario. |
| `--dataset-file`| `String` | `"TinyStories-train.txt"` | Nombre del archivo a extraer del dataset de HF. |
| `--vocab-size` | `usize` | `8000` | Tamaño total de la tabla de vocabulario resultante del tokenizador. |

---

## 🔮 5. Parámetros del Subcomando `generate`
Realiza inferencia autorregresiva de texto usando el modelo cargado en `model_output/`.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `<prompt>` | `String` | *Requerido* | Texto semilla inicial a partir del cual el modelo continuará la generación. |
| `--temp` | `f32` | `0.7` | Temperatura de muestreo. Controla la aleatoriedad de la predicción. |
| `--top-k` | `usize` | `40` | Muestreo Top-K. Limita la selección a los $K$ tokens más probables. |
| `--top-p` | `f32` | `0.9` | Muestreo Top-P (Nucleus Sampling). Limita la selección a tokens cuya probabilidad acumulada alcanza $P$. |
| `--min-p` | `f32` | `0.05` | Muestreo Min-P. Excluye tokens con probabilidad menor a $\text{min\_p} \times P_{\max}$. |
| `--rep-penalty` | `f32` | `1.1` | Penalización por repetición. Desincentiva tokens repetidos dividiendo sus logits. |
| `--max-tokens` | `usize` | `50` | Cantidad máxima de nuevos tokens a generar. |
| `--seed` | `Option<u64>` | `None` | Semilla aleatoria del generador Xorshift para reproducibilidad. |

---

## 📦 6. Parámetros del Subcomando `export`
Exporta los pesos a formato binario GGUF v3 para inferencia externa.

| Parámetro | Tipo | Por Defecto | Descripción Técnica / Función |
| :--- | :--- | :--- | :--- |
| `--model-name` | `String` | `"kuro-neko-qwen"` | Nombre base del archivo `.gguf` que se guardará en `exported/`. |

