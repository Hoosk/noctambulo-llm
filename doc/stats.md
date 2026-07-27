# 📊 Kuro-Neko — Estadísticas del Modelo (`stats`)

El subcomando `stats` proporciona un diagnóstico exhaustivo sobre la arquitectura del modelo, la cantidad exacta de parámetros entrenables, el consumo de memoria en precisión simple FP32, la intensidad computacional estimada en FLOPs por token y la compresión de VRAM lograda mediante **Grouped Query Attention (GQA)**.

---

## ⚙️ Uso del Subcomando

El subcomando `stats` **no requiere ningún parámetro** en la línea de comandos:

```bash
cargo run --release -- stats
```

---

## 📂 Origen de los Datos de Configuración

Al ejecutarse, el subcomando realiza el siguiente procedimiento:
1. Busca el archivo de configuración del modelo en `model_output/config.json`.
2. Si el archivo existe, lee los hiperparámetros del modelo entrenado (`n_embd`, `n_head`, `n_kv_head`, `n_layer`, `vocab_size`, `n_ctx`, `attn_softcap`).
3. Si el archivo no existe, utiliza los valores por defecto del sistema (`vocab_size=8000`, `n_embd=256`, `n_head=4`, `n_kv_head=4`, `n_layer=4`, `n_ctx=256`).

---

## 📐 Métricas Calculadas y Fórmulas Matemáticas

### 1. Desglose de Parámetros por Bloque Transformer
Para una dimensión de embedding $D = \text{n\_embd}$, número de capas $L = \text{n\_layer}$, vocabulario $V = \text{vocab\_size}$, cabezas $H = \text{n\_head}$ y cabezas KV $H_{\text{kv}} = \text{n\_kv\_head}$:

- **Dimensión por Cabeza**: $d_k = D / H$.
- **Dimensión de Key/Value**: $D_{\text{kv}} = H_{\text{kv}} \times d_k$.
- **Dimensión Oculta SwiGLU** (Alineada a múltiplos de 256):
  $$D_{\text{hidden}} = \left\lceil \frac{\lfloor 8D / 3 \rfloor}{256} \right\rceil \times 256$$
- **Parámetros por Capa/Bloque**:
  - Proyección $Q$ (con bias): $D \times D + D$
  - Proyecciones $K, V$ (con bias): $2 \times (D \times D_{\text{kv}} + D_{\text{kv}})$
  - Proyección $O$ (sin bias): $D \times D$
  - Capas SwiGLU MLP (Gate, Up, Down): $2 \times (D \times D_{\text{hidden}}) + (D_{\text{hidden}} \times D)$
  - RMSNorms de la capa: $2 \times D$
- **Total de Parámetros**:
  $$N_{\text{total}} = (V \times D) + L \times \text{Parámetros\_por\_Bloque} + D_{\text{norm\_final}}$$

---

### 2. Consumo de Memoria en FP32
Cada parámetro en precisión flotante de 32 bits (FP32) requiere 4 bytes de almacenamiento:
$$\text{Memoria FP32 (MB)} = \frac{N_{\text{total}} \times 4}{1024 \times 1024}$$

---

### 3. Operaciones de Punto Flotante por Token (FLOPs/Token)
En un pase hacia adelante (*forward pass*) autorregresivo, cada parámetro participa aproximadamente en 2 operaciones de punto flotante (una multiplicación y una adición):
$$\text{FLOPs por Token} = 2 \times N_{\text{total}}$$

---

### 4. Ahorro de Memoria de Caché KV con GQA
Mide el porcentaje de reducción en el tamaño de los tensores de Clave ($K$) y Valor ($V$) en VRAM gracias al uso de Grouped Query Attention en lugar de Multi-Head Attention convencional:
$$\text{Ahorro GQA (\%)} = \left(1 - \frac{H_{\text{kv}}}{H}\right) \times 100\%$$

---

## 🖥️ Ejemplo de Salida del Comando

Al ejecutar `cargo run --release -- stats` con la configuración base, el subcomando genera un informe formateado en consola como el siguiente:

```text
===============================================================================
🐈‍⬛ Kuro-Neko 2026 Model Architecture Statistics
===============================================================================
  Vocab Size       : 8000 tokens
  Context Window   : 256 tokens
  Embedding Dim    : 256
  Hidden SwiGLU Dim: 768 (aligned to 256)
  Query Heads      : 4
  KV Heads (GQA)   : 2 (Ratio 1:2)
  Layers           : 4
  Softcapping Cap  : Disabled
-------------------------------------------------------------------------------
  Total Parameters : 3481088 (3.48 Million)
  Model Size (FP32): 13.28 MB
  FLOPs per Token  : 0.01 GFLOPs
  GQA Cache Saving : 50.0% KV-cache VRAM reduction
===============================================================================
```
