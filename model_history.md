# 📖 Historia y Tutorial de Desarrollo: Kuro-Neko (kuro-neko-llm)

Este documento registra de forma objetiva la evolucion tecnica de **Kuro-Neko**, desde un modelo de juguete básico hasta un motor de entrenamiento e inferencia de última generacion (estándar 2026) escrito integramente en **Rust**, acelerado por GPU con **Burn 0.21** y **WGPU**, y exportable a formato binario nativo **GGUF v3**.

---

## 📑 Indice de Contenidos

1. [Capitulo 1: La Semilla y el Punto de Partida](#capitulo-1-la-semilla-y-el-punto-de-partida)
2. [Capitulo 2: Tokenizacion Paralela en Streaming (Zero-Copy)](#capitulo-2-tokenizacion-paralela-en-streaming-zero-copy)
3. [Capitulo 3: Diagnostico y Eliminacion de Cuellos de Botella GPU](#capitulo-3-diagnostico-y-eliminacion-de-cuellos-de-botella-gpu)
4. [Capitulo 4: Escalado de Datos, Chinchilla y Parametrizacion CLI](#capitulo-4-escalado-de-datos-chinchilla-y-parametrizacion-cli)
5. [Capitulo 5: El Salto a la Arquitectura 2026 (LLaMA / Qwen Standard)](#capitulo-5-el-salto-a-la-arquitectura-2026-llama--qwen-standard)
6. [Capitulo 6: Exportador Binario Nativo GGUF v3](#capitulo-6-exportador-binario-nativo-gguf-v3)
7. [Guia Rapida y Tutorial de Uso](#guia-rapida-y-tutorial-de-uso)
8. [🐣 Apendice For Dummies: Conceptos Explicados sin Modismos Tecnicos](#apendice-for-dummies-conceptos-explicados-sin-modismos-tecnicos)

---

## Capitulo 1: La Semilla y el Punto de Partida

### El Objetivo
Construir un Modelo de Lenguaje Autorregresivo (LLM) Causal desde cero utilizando **Rust** para garantizar seguridad de memoria sin recolector de basura (GC), y el framework **Burn** para aprovechar la aceleracion GPU multiplataforma (Vulkan/Metal/DX12) mediante **WGPU**.

### La Arquitectura Inicial (GPT-2 Baseline)
En las primeras etapas, el proyecto implemento la arquitectura clasica de GPT-2 (2019):
- **Token Embeddings (`wte`)** + **Positional Embeddings (`wpe`)**: Matrices fijas de embeddings aprendidos.
- **LayerNorm Pre-LN**: Normalizacion clasica con calculo de media, varianza y parametros de bias.
- **Atencion Causal Multi-Cabeza (MHA)**: Atencion producto-escalar con mascara triangular para impedir la vision hacia el futuro.
- **MLP Feed-Forward con GELU**: Proyeccion $N \to 4N \to N$ con activaciones `GELU`.

---

## Capitulo 2: Tokenizacion Paralela en Streaming (Zero-Copy)

### El Problema
Al cargar datasets grandes (como `TinyStories` de ~1.8 GB de texto plano), la primera implementacion intentaba tokenizar todo el archivo cargandolo completamente en RAM de un solo golpe, alcanzando picos de consumo de hasta **20 GB de memoria RAM** y velocidades de solo ~2 MB/s.

### La Solucion
1. **Streaming multihilo con `encode_batch`**: Se implemento `encode_file_streaming` en `src/data.rs` utilizando `BufReader` con un buffer de 2 MB y enviando lotes de 8,192 lineas a todos los cores del CPU.
2. **Optimizacion Zero-Copy en Rust**:
   - Reemplazo de la duplicacion de Strings por transferencia de propiedad a coste cero usando `std::mem::take(&mut line_batch)`.
   - Recorte de espacios en blanco in-place (`trim()`).
   - Resultado: Velocidad incrementada a **11-40 MB/s** y consumo de memoria RAM estabilizado en solo **~2.4 GB**.

---

## Capitulo 3: Diagnostico y Eliminacion de Cuellos de Botella GPU

Durante el desarrollo de los primeros entrenamientos, la GPU se bloqueaba o avanzaba a una velocidad penosa (~1 batch/s). Se realizo una auditoria profunda del codigo identificando tres fallos graves de rendimiento:

### 1. El Stall GPU $\to$ CPU por Batch
- **Fallo**: En cada iteracion del bucle de entrenamiento se llamaba a `.into_data().to_vec::<f32>()` para leer el escalar de Loss.
- **Impacto**: Forzaba a la canalizacion (pipeline) de la GPU a detenerse por completo en cada lote para hacer un readback sincrono PCIe a la CPU.
- **Solucion**: Acumulacion de tensores de Loss directamente en la memoria de la GPU usando `.detach()` (para evitar retencion de grafos de autograd) y sincronizacion a la CPU unicamente en los intervalos de log programados.

### 2. La Bomba de Memoria: One-Hot de 262 MB por Paso
- **Fallo**: Para calcular la perdida con `cross_entropy_with_logits`, los targets se convertian a una matriz `one_hot` de forma `[batch_size * n_ctx, vocab_size]`.
- **Impacto**: Con `batch_size = 128` y `n_ctx = 64`, se creaba una matriz de $8,192 \times 8,000 = 65.5 \text{ millones de floats}$ (**262 Megabytes**) de asignacion dinamica en VRAM **por cada paso de entrenamiento**.
- **Solucion**: Reemplazo por calculo esparso de entropia cruzada:
  $$\text{loss} = -\text{mean}(\text{gather}(\text{log\_softmax}(\text{logits}), \text{targets}))$$
  El consumo de memoria bajo de **262 MB a 32 KB** por batch (**8,000 veces menos asignacion**).

### 3. Mascara Causal Directa
- **Fallo**: En cada capa y cada paso se generaban tensores de posicion `arange` y se ejecutaba una inversion booleana `mask.equal_elem(false)`.
- **Solucion**: Mascara triangular superior directa `arange.unsqueeze(0).greater(arange.unsqueeze(1))` sin asignaciones intermedias.

---

## Capitulo 4: Escalado de Datos, Chinchilla y Parametrizacion CLI

### Entendiendo la Escala de Entrenamiento (Chinchilla Laws)
Para un modelo pequeño de **~7M a 30M de parametros**, intentar entrenarlo sobre 447 millones de tokens multiplicados por 3 epocas (1.34 mil millones de tokens) provocaba tiempos de entrenamiento artificialmente altos (dias) sin mejora en la capacidad del modelo.

### Cambios Aplicados
- **Paso de ventana sin solapamiento (`step = n_ctx`)**: Eliminacion del solapamiento artificial que duplicaba las secuencias.
- **Limite Optimo de Tokens (`--max-tokens 50000000`)**: Seleccion de 50 millones de tokens, el volumen matematicamente ideal para saturar un modelo de este tamaño.
- **Parametrizacion Completa por CLI**: Implementacion de flags en `src/main.rs` para controlar `--batch-size`, `--epochs`, `--max-tokens`, `--step`, `--n-ctx`, `--n-embd`, `--n-head`, `--n-layer` y `--lr`.

---

## Capitulo 5: El Salto a la Arquitectura 2026 (LLaMA / Qwen Standard)

Con la base estabilizada, se refactorizo la arquitectura del modelo en `src/model.rs` para alinearse con el estándar de los LLMs modernos de 2026 (como LLaMA 3/4, Qwen 2.5/3 y Gemma 2/3):

| Componente | GPT-2 Baseline (Anterior) | Kuro-Neko 2026 (Actual) | Beneficio Tecnico |
|:---|:---|:---|:---|
| **Posiciones** | Learned Embeddings (`wpe`) | **RoPE (Rotary Position Embeddings)** | Elimina la matriz fija `wpe`. Permite contexto dinámico en inferencia. |
| **Normalizacion** | `LayerNorm` (con bias) | **RMSNorm** | Normalizacion por media cuadratica sin bias. ~15% mas rapido en GPU. |
| **Activacion MLP** | `GELU` | **SwiGLU FFN** | Red gating triple ($\text{SiLU}(X W_g) \odot X W_u$) $W_d$. Mayor capacidad de representacion. |
| **Inferencia** | Muestreo Aleatorio Basico | **Top-K + Top-P + Repetition Penalty** | Elimina bucles repetitivos e historias incoherentes. |
| **Scheduler** | Tasa de aprendizaje fija | **Cosine LR + Linear Warmup** | Calibracion suave del gradiente para alcanzar perdidas (loss) mas bajas. |

---

## Capitulo 6: Exportador Binario Nativo GGUF v3

Se creo un serializador binario puro en Rust (`src/export.rs`) que escribe el modelo directamente en formato **GGUF v3**:

1. **Estructura de Cabecera**: Escribe la firma magica `b"GGUF"` y la version `3u32`.
2. **Metadatos de Arquitectura**: Configura `"general.architecture" = "llama"`, `"general.name" = "kuro-neko-2026"`, e inserta el numero de capas, cabezas y longitud de contexto.
3. **Mapeo de Tensores LLaMA**: Mapea los tensores de Burn a la nomenclatura estandar de llama.cpp:
   - `token_embd.weight`
   - `blk.N.attn_norm.weight`
   - `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`, `blk.N.attn_output.weight`
   - `blk.N.ffn_norm.weight`
   - `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`
   - `output_norm.weight`
   - `output.weight`
4. **Alineacion a 32 Bytes**: Rellena con ceros los espacios entre tensores para cumplir con el requisito de alineacion SIMD/GPU de **LM Studio** y **llama.cpp**.

---

## 🚀 Guia Rapida y Tutorial de Uso

### 1. Entrenar el Modelo
```bash
# Entrenamiento optimizado (50M tokens, batch 128, 3 epocas)
cargo run --release train --dataset roneneldan/TinyStories

# Prueba rapida de 2 minutos (10M tokens)
cargo run --release train --dataset roneneldan/TinyStories --max-tokens 10000000
```

### 2. Generar Texto
```bash
cargo run --release generate "Once upon a time" --temp 0.7 --top-k 40 --top-p 0.9 --rep-penalty 1.1
```

### 3. Exportar a GGUF para LM Studio / Ollama
```bash
cargo run --release -- export --model-name kuro-neko-qwen
```
El archivo `.gguf` generado en `exported/` se puede arrastrar directamente a **LM Studio** o cargar en **llama.cpp**.

---

## 🐣 Apendice For Dummies: Conceptos Explicados sin Modismos Tecnicos

Si todo lo anterior suena a chino mandarín, esta seccion explica cada concepto con analogias sencillas de la vida real.

### 1. ¿Que es exactamente un LLM?
Es un **autocorrector de teclado de movil**, pero con esteroides. El autocorrector del movil mira las ultimas 2 o 3 palabras y adivina la siguiente. Un LLM mira cientos o miles de palabras anteriores, entiende las relaciones entre ellas y predice la palabra que tiene mas sentido despues.

### 2. ¿Que son los Tokens y la Tokenizacion?
La Inteligencia Artificial **no sabe leer letras ni palabras enteras**, solo entiende numeros. 
- **Tokenizar** es pasar el texto por unas "tijeras inteligentes" que cortan las palabras en trocitos (tokens). Por ejemplo, la palabra `desafortunadamente` podria cortarse en `des-`, `afortunada-`, `mente`.
- Cada trocito tiene un numero asignado en un diccionario. La IA solo ve listas de numeros como `[482, 1902, 53]`.

### 3. ¿Que es la Ventana de Contexto (`n_ctx`)?
Es la **memoria a corto plazo de la IA**. 
- Si el contexto es de 64 tokens, la IA solo puede mirar las ultimas ~40 palabras para saber de que va la conversacion. Todo lo anterior se le olvida por completo.
- Si subimos el contexto a 512 o 2048, la IA puede recordar un capitulo entero y no cambiara el nombre del protagonista a mitad del cuento.

### 4. ¿Por que RoPE es mejor que la posicion antigua (`wpe`)?
- **El metodo antiguo (`wpe`)**: Era como los asientos numerados de un cine. La IA tenia una tabla fija que decia "Casilla 1, Casilla 2... hasta la Casilla 64". Si le pedias sentarse en la casilla 65, la IA entraba en panico porque ese asiento no existia en su mapa.
- **RoPE (Metodo 2026)**: En lugar de asientos fijos, las palabras "giran" como las agujas de un reloj segun donde esten. La IA no memoriza casillas, sino la distancia angular entre palabras. Esto permite ampliar el cine de 64 a 4,000 asientos sin tener que rehacer la IA.

### 5. ¿Que es la Perdida (Loss) y por que queremos que baje?
El **Loss es la nota de fallo en el examen de la IA**:
- Al principio del entrenamiento, la IA dice tonterias al azar y saca un fallo altisimo (**Loss = 9.0**).
- Conforme lee millones de frases y ajusta sus neuronas, empieza a acertar las palabras y la nota de fallo baja (**Loss = 2.8**).
- **Cuanto mas bajo sea el Loss, mejor y mas coherente habla la IA**.

### 6. ¿Que era la "Bomba de Memoria One-Hot" de 262 MB y por que la quitamos?
Imagina que para decirle a alguien la palabra numero 4, en lugar de decirle simplemente "el numero 4", imprimes un diccionario entero de **8,000 paginas en blanco** y solo pones una cruz `X` en la pagina 4.
- Eso hacia el código anterior: imprimia 8,000 paginas en blanco en la memoria de la tarjeta grafica por cada paso de entrenamiento (**262 Megabytes desperdiciados por lote**).
- Al cambiar a **perdida esparsa**, pasamos a enviar directamente el numero 4 (**solo 32 Kilobytes**). La tarjeta grafica paso de ahogarse a volar.

### 7. ¿Que son RMSNorm y SwiGLU?
- **RMSNorm**: Es como quitarle el equipaje y los asientos de atras a un coche de carreras para hacerlo mas ligero. Elimina calculos innecesarios (la media y los biases) para que la GPU procese mas rapido.
- **SwiGLU**: Es como ponerle un turbocompresor al motor. Es un tipo de "neurona" con un filtro de paso (gating) que ayuda a la IA a captar matices complejos del lenguaje con menos neuronas totales.

### 8. ¿Que hacen Top-K, Top-P y Repetition Penalty al generar texto?
Cuando la IA inventa una historia, no elige siempre la palabra mas probable (si lo hiciera, seria aburridisima y repetitiva). Elige probando suerte entre varias opciones, pero usamos tres filtros de seguridad:
- **Top-K (ej. 40)**: "De las 8,000 palabras del diccionario, tira a la basura las 7,960 mas raras y elige solo entre las 40 mejores".
- **Top-P / Nucleus (ej. 0.9)**: "Quedate solo con las palabras que juntas sumen el 90% de probabilidad de tener sentido".
- **Repetition Penalty**: "Si la IA empieza a repetir *'habia una vez una vez una vez'*, le damos un pequeño pellizco para obligarla a cambiar de palabra".

### 9. ¿Que es un archivo GGUF y por que nos importa?
Es el **formato de archivo universal para IA** (equivalente a lo que es el `.mp3` para las canciones o el `.pdf` para los documentos).
- Empaqueta todas las neuronas de la IA en un solo archivo binario.
- Permite que programas de escritorio faciles de usar como **LM Studio** o **llama.cpp** puedan abrir tu modelo en Windows, Mac o Linux arrastrando y soltando el archivo, sin necesidad de saber programar.

---

## Capitulo 7: La Arquitectura Definitiva (Qwen2.5 / Qwen3 Standard)

### El Diagnostico de Calidad y Solucion
1. **La Trampa de Inicializacion del Weight Tying**:
   - Al activar Weight Tying, la libreria Burn inicializaba la matriz compartida con distribución normal estándar ($\sigma = 1.0$).
   - Con $\sigma = 1.0$, el producto matricial de salida generaba logits gigantescos ($\pm 300$), saturando el softmax y destruyendo los gradientes.
   - **Solución Qwen**: Fijar la inicialización de embeddings a la distribución **$\mathcal{N}(0, 0.02)$**, manteniendo los logits en el rango controlado de $[-2, +2]$ y permitiendo la convergencia fluida.

2. **Atencion QKV Bias**:
   - Siguiendo la especificación oficial de **Qwen2.5 / Qwen3** (Alibaba), las proyecciones $Q, K, V$ ahora incluyen **sesgo (bias)** (`with_bias(true)`), mientras que las proyecciones de salida y MLP permanecen sin sesgo. Esto estabiliza el flujo de gradientes en modelos de tamaño reducido.

3. **Parametrización Oficial Qwen**:
    - **RMSNorm Épsilon**: $\epsilon = 10^{-6}$.
    - **RoPE Base Frequency**: $\theta = 10.000$ (para contextos estándar / optimizado para n_ctx <= 8K).
    - **Weight Decay**: $0.1$ para prevenir overfitting en entrenamientos de múltiples épocas.

4. **Compatibilidad GGUF v3**:
   - El exportador serializa los tensores de bias de atención (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) y exporta la matriz de salida ligada, garantizando compatibilidad total con `llama.cpp` y LM Studio.

---

## Capitulo 8: Optimizaciones Avanzadas (GQA, Min-P, Softcapping & Stats CLI)

### 1. Atención por Grupos (GQA - Grouped Query Attention)
- **El Problema MHA**: En Multi-Head Attention estándar, cada cabeza de consulta tiene su propia cabeza de clave y valor ($K, V$), multiplicando el consumo de memoria VRAM por cada token generado.
- **La Solución GQA**: Varias cabezas de consulta comparten la misma cabeza $K, V$ (`n_kv_head < n_head`). Esto reduce el consumo de memoria KV-Cache hasta un **75%**, permitiendo mayores tamaños de lote e inferencia $4\times$ más rápida.

### 2. Muestreo Matemático Min-P
- A diferencia de Top-P (que usa un corte estático acumulado), **Min-P** calcula un umbral dinámico basado en la probabilidad del token más probable ($T = \text{min\_p} \times P_{\max}$).
- Cuando el modelo está 99% seguro de una respuesta, Min-P recorta agresivamente tokens ruidosos. Cuando el modelo está incierto, Min-P preserva alternativas creativas.

### 3. Logit Softcapping (Gemma 4 Math)
- Aplica una función de acotamiento suave a los scores de atención:
  $$\text{scores} = \text{cap} \times \tanh\left(\frac{Q K^T}{\sqrt{d_k} \times \text{cap}}\right)$$
- Garantiza que ninguna exponencial en softmax desborde la precisión flotante de la GPU en secuencias de gran longitud.

### 4. Subcomando `stats`
- Permite calcular al instante la huella matemática completa del modelo: parámetros totales, tamaño FP32 en megabytes, FLOPs por token y ratio de ahorro en VRAM por GQA.
