# 🏋️ Kuro-Neko — Entrenamiento (`train`)

El subcomando `train` es el núcleo del motor **Kuro-Neko**. Permite entrenar desde cero un modelo de lenguaje autorregresivo Causal Transformer (arquitectura Qwen2.5 / Qwen3) acelerado completamente por GPU a través del backend **WGPU** del framework **Burn 0.21**.

Durante el entrenamiento, Kuro-Neko procesa conjuntos de datos en formato de texto plano (procedentes de **Hugging Face Hub** o de un archivo local), codifica el texto con el tokenizador BPE en flujo paralelo (*streaming*), construye ventanas deslizantes de secuencias de contexto (`n-ctx`) y ejecuta el bucle de optimización AdamW en la GPU.

---

## ⚙️ Tabla Completa de Parámetros CLI

El subcomando `train` acepta **23 parámetros** de configuración CLI:

| Flag | Tipo | Valor por Defecto | Descripción Detallada |
| :--- | :--- | :--- | :--- |
| `--dataset <repo>` | `Vec<String>` (separados por coma) | `[]` | Identificador(es) del repositorio de Hugging Face (ej. `roneneldan/TinyStories`). Admite múltiples repositorios separados por comas para entrenamiento combinado. |
| `--file <path>` | `Option<String>` | `None` | Ruta a un archivo de texto local en el sistema de archivos para usar como corpus de entrenamiento. |
| `--dataset-file <filename>` | `String` | `TinyStories-train.txt` | Nombre del archivo objetivo a descargar e inspeccionar dentro del repositorio de Hugging Face. |
| `--preset <name>` | `Option<String>` | `None` | Preset de arquitectura (`nano`, `tiny`, `small`, `base`); los flags explícitos de dimensión tienen prioridad. |
| `--dtype <name>` | `String` | `f32` | Precisión de cómputo (`f32` o `f16`). `f16` usa el `CubeBackend` sin fusión (burn-fusion 0.21 crashea con streams half en Vulkan) y suele ser un 2–3× más rápido; puede desestabilizar si el LR es alto. `bf16` se rechaza con un error claro porque WGSL no lo soporta. Los checkpoints se guardan en f32. |
| `--batch-size <n>` | `usize` | `128` | Número de secuencias por lote (*batch*) de entrenamiento. Un tamaño mayor aprovecha mejor la GPU y acelera la convergencia, pero requiere más VRAM. |
| `--epochs <n>` | `usize` | `3` | Número de pasadas completas a través del conjunto de datos procesado. |
| `--max-tokens <n>` | `usize` | `100000000` (100M) | Número máximo de tokens a cargar desde el conjunto de datos. Un valor de `0` desactiva el límite y carga el dataset completo. El presupuesto Chinchilla de referencia es ~20 tokens por parámetro. |
| `--step <n>` | `Option<usize>` | `None` (`= n-ctx`) | Paso de la ventana deslizante en tokens. Controla la superposición entre secuencias consecutivas. Si se omite, equivale a `n-ctx` (sin superposición). |
| `--n-ctx <n>` | `usize` | `256` | Longitud de la ventana de contexto del modelo en tokens. Determina el tamaño máximo de secuencia que el modelo puede observar a la vez. |
| `--n-embd <n>` | `usize` | `256` | Dimensión del espacio de embedding y de los estados ocultos. Valores más grandes incrementan la capacidad de representación. |
| `--n-head <n>` | `usize` | `4` | Número de cabezas de atención para Query ($Q$). La dimensión de embedding `n-embd` debe ser divisible entre `n-head`. |
| `--n-kv-head <n>` | `Option<usize>` | `None` (`= n-head`) | Número de cabezas de Key/Value ($K, V$) para Grouped Query Attention (GQA). `None` o `0` activa Multi-Head Attention (MHA). Un valor menor reduce el consumo de VRAM en la caché KV. |
| `--n-layer <n>` | `usize` | `4` | Número de bloques Transformer apilados en el modelo. Mayor número de capas permite aprender patrones jerárquicos más profundos. |
| `--lr <float>` | `f64` | `0.0003` (3e-4) | Tasa de aprendizaje pico (*peak learning rate*). Se ajusta dinámicamente mediante un schedule coseno con *warmup*. |
| `--dropout <float>` | `f64` | `0.1` | Probabilidad de dropout aplicada a las probabilidades de atención y a las proyecciones residuales/MLP como regularización. |
| `--attn-softcap <float>` | `f32` | `0.0` | Umbral de softcapping para los scores de atención. `0.0` desactiva la función. Un valor como `50.0` aplica la fórmula estilo Gemma 4 para estabilizar gradientes. |
| `--val-split <float>` | `f32` | `0.1` | Fracción del corpus reservada para validación (cálculo de Loss y Perplexity por época). `0.0` desactiva la validación. |
| `--grad-accum <n>` | `usize` | `1` | Pasos de acumulación de gradientes por actualización de AdamW. El gradiente acumulado es la **media** de los micro-batches (equivalente a un batch más grande). |
| `--save-dir <path>` | `Option<String>` | `None` | Directorio donde guardar `checkpoint_epoch_N.bin` y `checkpoint_best.bin` al final de cada época. |
| `--resume <path>` | `Option<String>` | `None` | Checkpoint `.bin` (o su directorio) desde el que reanudar. Restaura pesos, momentos de AdamW, posición del schedule y estado del shuffler. `--epochs` pasa a significar **épocas adicionales**. |
| `--seq-warmup-steps <n>` | `usize` | `0` | Pasos iniciales de *Progressive Sequence Scaling*: el contexto crece de `n_ctx/4` hasta `n_ctx` durante el warmup. `0` lo desactiva. |
| `--early-stop-patience <n>` | `usize` | `0` | Detiene el entrenamiento tras `n` épocas consecutivas sin mejorar `val_loss`. `0` lo desactiva. |

---

## 📈 Algoritmo y Detalles de Implementación

### 1. Schedule de Learning Rate Cosine con Warmup

Kuro-Neko implementa un esquema de decaimiento coseno de la tasa de aprendizaje (*Cosine Annealing Learning Rate Schedule*) complementado con una fase inicial de calentamiento lineal (*Warmup*):

- **Pasos de Warmup**: Se calculan automáticamente como:
  $$\text{warmup\_steps} = \begin{cases} 0 & \text{si } \text{total\_steps} \le 10 \\ \min(1000, \lfloor \frac{\text{total\_steps}}{10} \rfloor) & \text{en otro caso} \end{cases}$$
- **LR Mínimo**: La tasa final de aprendizaje decae hasta un 10% del valor pico ($\text{min\_lr} = \text{lr} \times 0.1$).
- **Fase 1 (Warmup Lineal)**: Durante los primeros $\text{warmup\_steps}$, el LR se incrementa de forma estrictamente lineal desde 0 hasta `lr`.
- **Fase 2 (Decaimiento Coseno)**:
  $$\text{lr}_t = \text{min\_lr} + 0.5 \times (\text{lr} - \text{min\_lr}) \times \left(1 + \cos\left(\pi \times \frac{t - \text{warmup\_steps}}{\text{total\_steps} - \text{warmup\_steps}}\right)\right)$$

---

### 2. Gradient Clipping (Recorte de Gradientes)

Para prevenir la explosión de gradientes (*exploding gradients*) en secuencias largas o durante pasos con pérdidas elevadas, el optimizador aplica un recorte por norma $L_2$ estricto de **1.0** (`GradientClippingConfig::Norm(1.0)`). 

Si la norma euclídea del vector global de gradientes $g$ excede el valor $1.0$, los gradientes se reescalan según:
$$g \leftarrow g \times \frac{1.0}{\max(1.0, \|g\|_2)}$$

---

### 3. Optimizador AdamW y Weight Decay

Se utiliza el optimizador **AdamW** con las siguientes propiedades:
- **Decaimiento de Pesos (Weight Decay)**: Regulado a $WD = 0.1$.
- Separación explícita del decaimiento de pesos respecto a la actualización de los momentos del gradiente, evitando que los pesos importantes se degraden desproporcionadamente.

---

### 4. Mezcla de Datos por Época (Fisher-Yates Shuffling)

En cada época de entrenamiento, el conjunto de índices de las secuencias procesadas se reordena aleatoriamente mediante el algoritmo de **Fisher-Yates** impulsado por un generador de números pseudoaleatorios `XorshiftRng` determinista:

```rust
fn shuffle_indices(indices: &mut [usize], rng: &mut XorshiftRng) {
    let len = indices.len();
    for i in (1..len).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
        indices.swap(i, j);
    }
}
```
Esto garantiza que los lotes (*batches*) de cada época contengan diferentes combinaciones de secuencias, evitando que el modelo memorice el orden del corpus.

---

### 5. Cálculo de Pérdida Esparsa en GPU (Sparse NLL Loss)

La función de pérdida para el modelado de lenguaje autorregresivo se calcula directamente en la GPU usando log-softmax y la operación `gather`:

1. **Log-Softmax**:
   $$\text{log\_probs} = \text{log\_softmax}(\text{logits\_flat}, \text{dim}=1)$$
2. **Gather**: Selecciona los log-probabilidades correspondientes a las etiquetas objetivo (*targets*):
   $$\text{target\_log\_probs} = \text{log\_probs}.\text{gather}(1, \text{targets\_flat})$$
3. **Pérdida NLL**:
   $$\text{Loss} = - \text{mean}(\text{target\_log\_probs})$$

> [!IMPORTANT]
> **Acumulación Asíncrona en GPU**: Para evitar los cuellos de botella por sincronización CPU $\to$ GPU en cada paso de batch, la pérdida calculada se desengancha del grafo de autodiferenciación (`detach()`) y se acumula en un tensor residente en la memoria de la GPU (`accum_loss_tensor`). Únicamente se lee hacia la CPU en los intervalos de reporte configurados.

---

### 6. Acumulación Real de Gradientes (`--grad-accum`)

Con `--grad-accum N` cada actualización de AdamW agrupa $N$ micro-batches. La pérdida de cada micro-batch se divide por el tamaño real de la ventana ($N$, o menos en la ventana parcial final de cada época) y los gradientes se suman con `burn::optim::GradientsAccumulator`:

$$\nabla_\theta \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i$$

Esto garantiza que acumular $N$ micro-batches de tamaño $B$ sea **matemáticamente equivalente** a un solo batch de tamaño $N \times B$ (linealidad del gradiente), que es exactamente lo que verifica el test `test_accumulated_gradients_match_full_batch`.

---

### 7. Estado de Entrenamiento Persistente y Reanudación

Al final de cada época (y al detener el entrenamiento) Kuro-Neko guarda, junto a los pesos:

- `train_state.json`: `global_step`, `total_steps`, `epochs_done`, `best_val_loss` y el estado interno del shuffler.
- `optimizer_state.bin`: el record de AdamW (momentos de primer y segundo orden) serializado con `Optimizer::to_record`.

Al reanudar con `--resume <checkpoint>` se restauran los pesos **y** ambos sidecars, por lo que el schedule coseno continúa desde el paso guardado, los momentos de AdamW no se pierden y el orden de las secuencias sigue donde estaba. `--epochs` en un resume indica épocas **adicionales**.

Si no existe `train_state.json` (checkpoints previos), el resume es retrocompatible: se cargan solo los pesos y se avisa de que el estado de AdamW arranca de cero.

### 8. Parada Segura (Stop / Ctrl+C)

Detener un entrenamiento desde el servidor web o con `Ctrl+C` (CLI) no descarta el trabajo: antes de salir se guarda `checkpoint_interrupted.bin` junto con `train_state.json` y `optimizer_state.bin`, y se emite el evento `Checkpoint` con `kind = "interrupted"`. El servidor web además usa `with_graceful_shutdown` y espera hasta 30 segundos a que el worker termine de persistir.

### 9. Presupuesto de Tokens y Early Stopping

- **Aviso Chinchilla**: si `--max-tokens` queda por debajo de ~20 tokens por parámetro, el entrenador emite un aviso al inicio (p. ej. un modelo de 5.5M params pide ~110M tokens). Usa `--max-tokens 0` para entrenar con el dataset completo.
- **Early stopping**: con `--early-stop-patience N`, el bucle termina cuando `val_loss` no mejora durante `N` épocas; al salir se guardan los pesos y el estado como en una parada normal.
- **Métricas de validación**: la validación evalúa **todo** el split (ya no hay tope fijo de 100 batches) y reporta Loss, PPL y BPB. El BPB (`loss / ln2 × tokens/byte`) permite comparar modelos entrenados con tokenizers distintos; no está disponible para datasets `.bin` (no conservan el tamaño en bytes).

---

## 💻 Ejemplos Prácticos de Entrenamiento

### Ejemplo 1: Entrenamiento Estándar en TinyStories (Configuración por Defecto)
Entrena sobre el dataset TinyStories con 10M de tokens, batch size de 128 y 3 épocas:
```bash
cargo run --release -- train --dataset roneneldan/TinyStories
```

### Ejemplo 2: Entrenamiento Rápido de Prueba con GQA (Grouped Query Attention)
Entrenamiento ligero con 1 millón de tokens y ratio GQA 2:1 (`n-head=4`, `n-kv-head=2`):
```bash
cargo run --release -- train \
  --dataset roneneldan/TinyStories \
  --max-tokens 1000000 \
  --n-head 4 \
  --n-kv-head 2 \
  --batch-size 64 \
  --epochs 1
```

### Ejemplo 3: Entrenamiento sobre Archivo Local Personalizado
Entrena utilizando un archivo de texto local con ventana de contexto extendida a 512 tokens y superposición de ventana deslizante (`step=256`):
```bash
cargo run --release -- train \
  --file ./mis_documentos.txt \
  --n-ctx 512 \
  --step 256 \
  --n-embd 384 \
  --n-layer 6 \
  --lr 0.0005
```

### Ejemplo 4: Configuración Avanzada con Softcapping y Múltiples Dataset Repos
Entrena combinando dos repositorios de Hugging Face, con Softcapping de atención activado en `50.0` y sin límite de tokens (`max-tokens=0`):
```bash
cargo run --release -- train \
  --dataset roneneldan/TinyStories,wikitext \
  --max-tokens 0 \
  --n-embd 512 \
  --n-head 8 \
  --n-kv-head 2 \
  --n-layer 8 \
  --attn-softcap 50.0 \
  --dropout 0.05
```
