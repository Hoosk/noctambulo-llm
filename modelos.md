# 📊 Análisis y Comparativa de Arquitecturas LLM / SLM (2026)

Este documento contiene el análisis técnico comparativo de los principales modelos de lenguaje de código abierto de la generación **2026** (Gemma 4, Qwen 3.5/3.8, DeepSeek V4 Pro, Kimi K2.7, GLM-5.2 y LLaMA 3.2/4).

---

## 📊 Tabla Comparativa de Arquitecturas LLM/SLM (2026)

| Modelo | Creador / Fecha | Arquitectura | Parámetros (Totales / Activos) | `tie_word_embeddings` | Mecanismo de Atención | Normalización / Activación | Ventana Contexto | Licencia |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Gemma 4** | Google DeepMind *(Abril 2026)* | Híbrida (Dense & MoE) | E2B – 31B (MoE: 26B/3.8B) | **`true`** *(E2B/E4B)* **`false`** *(31B)* | **GQA** (Grouped Query) | RMSNorm + GeGLU | **256k** | Apache 2.0 |
| **Qwen 3.5 / 3.8** | Alibaba Cloud *(Julio 2026)* | Dense & MoE | 0.5B – 397B (Max: 2.4T MoE) | **`true`** *(0.5B-3B)* **`false`** *(7B+)* | **GQA + QKV Bias** | RMSNorm + SwiGLU | **1M** | Apache 2.0 / Token Plan |
| **DeepSeek V4 Pro** | DeepSeek *(Abril 2026)* | Sparse MoE | 1.6T / 49B activos | **`false`** | **CSA** (Compressed Sparse Attention) | RMSNorm + SwiGLU | **1M** | MIT |
| **Kimi K2.7 Code** | Moonshot AI *(Junio 2026)* | Sparse MoE | 1T / 32B activos | **`false`** | **MLA** (Multi-head Latent Attention) | RMSNorm + SwiGLU | **256k** | Modified MIT |
| **GLM-5.2** | Zhipu AI *(Junio 2026)* | Sparse MoE + IndexShare | 744B / ~36B activos | **`false`** | **Sparse MHA + IndexShare** | RMSNorm + SwiGLU | **1M** | MIT |
| **LLaMA 3.2 / 4** | Meta AI *(2025/2026)* | Dense & MoE | 1B – 405B | **`false`** | **GQA** (Grouped Query) | RMSNorm + SwiGLU | **128k - 1M** | LLaMA 3 Community |
| **Kuro-Neko Engine** | Rust + WGPU *(2026)* | SLM Causal Decoder | Configurable (5M – 3B) | **`true`** *(Tied N(0,0.02))* | **GQA + QKV Bias + Min-P + Softcap** | RMSNorm + SwiGLU | **256 - 128k** | MIT / Open Source |

---

## 🔑 Análisis de Tendencias de Arquitectura (2026)

### 1. `tie_word_embeddings` (Tied vs Untied)
- **Modelos Pequeños / SLMs (< 3B)**: **Usan `tie_word_embeddings: true`** (Gemma 4 E2B, Qwen 2.5/3 0.5B). Compartir los pesos entre la entrada y la salida ahorra hasta un 30% de VRAM en GPUs cliente móviles y de escritorio.
- **Modelos Gigantes / MoE (> 7B)**: **Usan `tie_word_embeddings: false`** (DeepSeek V4, GLM-5.2, Kimi K2.7). A gran escala, el ahorro relativo de VRAM por compartir embeddings es despreciable frente a la expresividad de tener matrices separadas.

### 2. Atención Evolucionada (Más allá del MHA estándar)
- **GQA (Grouped Query Attention)**: Estándar universal en Gemma 4, Qwen 3 y LLaMA 3.2.
- **MLA (Multi-head Latent Attention)**: Kimi K2.7 y DeepSeek comprimen las claves ($K$) y valores ($V$) en un espacio latente para reducir el consumo de KV-Cache a 1M de contexto.
- **QKV Bias**: Qwen 3 mantiene los sesgos en $Q, K, V$ para evitar la inestabilidad de gradientes en coma flotante fp16/bf16 en modelos reducidos.

### 3. Normalización y Activaciones Universales
- **RMSNorm** y **SwiGLU/GeGLU** se han convertido en el estándar absoluto sin excepción en **todos** los modelos insignia de 2026.
