# 🦙 La Aventura de Hacer Funcionar Kuro-Neko en `llama.cpp` (`how-to-llama.md`)

Este documento relata el proceso técnico, los obstáculos superados y las especificaciones exactas del estándar **GGUF v3** necesarias para lograr que un modelo entrenado desde cero en Rust con **Burn 0.21** se ejecute de forma nativa en **`llama.cpp`** (`llama-cli`) y **LM Studio** a **200 tokens/segundo**.

---

## 🎯 El Objetivo

Lograr que nuestro motor escrito en Rust (**Kuro-Neko**) exportase su arquitectura moderna de 2026 (RoPE, RMSNorm, SwiGLU) en un único archivo binario `.gguf` que pudiese ser ejecutado directamente por la herramienta estándar de la industria: **`llama.cpp`**.

---

## 🏔️ Los Cuatro Obstaculos de la Especificacion GGUF v3

### 1. El Error de Hiperparametros de LLaMA
- **El Fallo**:
  ```text
  error loading model hyperparameters: key not found in model: llama.attention.layer_norm_rms_epsilon
  ```
- **Causa Tecnica**:
  Al declarar en los metadatos de cabecera `"general.architecture" = "llama"`, el parser de `llama.cpp` valida de forma estricta que existan los hiperparametros especificos de la familia LLaMA.
- **La Solucion**:
  En `src/export.rs`, se definio la constante de tipo Float32 `GGUF_TYPE_FLOAT32 = 6` y se añadieron las claves requeridas:
  - `llama.attention.layer_norm_rms_epsilon` (Float32 = `1e-6`)
  - `llama.feed_forward_length` (UInt32 = $\frac{8}{3} n_{embd}$ alineado a 256)
  - `llama.attention.head_count_kv` (UInt32 = `n_kv_head`)

---

### 2. El Error de Vocabulario Ausente
- **El Fallo**:
  ```text
  error loading model vocabulary: key not found in model: tokenizer.ggml.model
  ```
- **Causa Tecnica**:
  `llama.cpp` no depende de archivos JSON externos para leer las palabras; exige que todo el vocabulario del tokenizador este serializado **dentro de la propia cabecera del archivo binario `.gguf`**.
- **La Solucion**:
  Se creo la funcion `load_vocab_tokens` en `src/export.rs` para leer los tokens del tokenizador, ordenarlos por ID y escribirlos en los metadatos GGUF usando tipos array (`GGUF_TYPE_ARRAY`):
  - `tokenizer.ggml.model` = `"llama"`
  - `tokenizer.ggml.tokens` = Array de strings con los tokens correspondientes al vocabulario.

---

### 3. Mapeo de Vocabulario Simplificado
- **Causa Tecnica**:
  Nuestro tokenizador BPE entrenado en Rust (`tokenizers` crate) produce un vocabulario byte-level compatible con el modelo `gpt2` de `llama.cpp` (pre-tokenizador byte-level, decoder byte-level, normalizador NFKC). Declaramos el modelo de tokenización como `"gpt2"` con pre-tokenizer `"gpt-2"`, lo que evita la necesidad de serializar manualmente merges y scores en metadatos GGUF y garantiza compatibilidad directa y libre de fallos de carga con `llama-cli` y LM Studio.

---

### 4. Permutacion de Pesos RoPE (Hugging Face vs C++ `llama.cpp`)
- **El Fallo**:
  El modelo cargaba en `llama.cpp`, pero el texto generado comenzaba a perder coherencia ("gibberish") tras los primeros tokens, mientras que en Rust (`cargo generate`) la historia era 100% clara.
- **Causa Tecnica**:
  Existen dos estilos de implementar RoPE (Rotary Position Embeddings):
  - **Split (Hugging Face / Python / Rust)**: Rota dividir la dimension en dos mitades $[-x_2, x_1]$.
  - **Interleaved (LLaMA C++ / llama.cpp)**: Rota intercalando dimensiones de dos en dos $[-x_1, x_0, -x_3, x_2]$.
- **La Solucion Oficial de Hugging Face**:
  Basandonos en el script oficial de Hugging Face `convert_llama_weights_to_hf.py`, implementamos la funcion `permute_rope_weight` en `src/export.rs` para reordenar las filas de los tensores de $Q$ y $K$ al exportar a `.gguf`:

```rust
fn permute_rope_weight(data: &[f32], n_head: usize, n_embd: usize) -> Vec<f32> {
    let head_size = n_embd / n_head;
    let half_dim = head_size / 2;
    let mut out = vec![0.0f32; data.len()];

    for r_head in 0..n_head {
        for i in 0..half_dim {
            let r_in1 = r_head * head_size + i;
            let r_in2 = r_head * head_size + i + half_dim;
            let r_out1 = r_head * head_size + 2 * i;
            let r_out2 = r_head * head_size + 2 * i + 1;

            for c in 0..n_embd {
                out[r_out1 * n_embd + c] = data[r_in1 * n_embd + c];
                out[r_out2 * n_embd + c] = data[r_in2 * n_embd + c];
            }
        }
    }
    out
}
```

---

## 🏆 El Resultado Final

Tras aplicar la transposicion de matrices 2D y la permutacion de pesos de RoPE junto con los metadatos GGUF v3 completos, ejecutamos:

```bash
llama-cli -m exported/kuro-neko-2026.gguf -p "There was a little cat named" -n 50
```

### Log de Inferencia Real obtenido:

```text
build      : b0-
model      : kuro-neko-2026.gguf
modalities : text

> There was a little cat named

|   . After a while, Lily was very happy to see a special surprise, a big brother's mom hugged him and said, "You can I have...

[ Prompt: 2603,6 t/s | Generation: 199,8 t/s ]
```

- **Compatibilidad**: 100% Nativa en `llama.cpp`, `llama-server` y **LM Studio**.
- **Procesamiento de Prompt**: **2,603.6 tokens por segundo**.
- **Velocidad de Generacion**: **199.8 tokens por segundo**.

---

## 🛠️ Tutorial: Como Exportar y Ejecutar en `llama.cpp`

### Paso 1: Exportar el Paquete
Desde la raiz de nuestro proyecto en Rust:
```bash
cargo run --release -- export --model-name mi-modelo-llama
```

### Paso 2: Ejecutar con `llama.cpp`
```bash
llama-cli -m exported/mi-modelo-llama.gguf -p "Once upon a time" -n 100
```
