# 🔍 Investigación y Panorama Global de LLMs en 2026 (`search.md`)

Este documento registra los datos reales de la industria de Inteligencia Artificial en **2026** (familias Llama 4, DeepSeek V4, Qwen 3.8-Max, Gemma 4) y la comparativa directa con la escala de nuestro motor **Kuro-Neko**.

---

## 🌐 1. El Panorama Global de Modelos en 2026

En 2026, la industria ha migrado masivamente hacia arquitecturas **Mixture-of-Experts (MoE)** para optimizar la velocidad de inferencia activando solo una pequeña parte del modelo por cada token generado.

### 🦙 Meta Llama 4
- **Llama 4 Scout**: 109B parametros totales (17B activos), ventana de contexto de **10 Millones de tokens**.
- **Llama 4 Maverick**: 400B parametros totales (17B activos), ventana de contexto de 1 Millon de tokens.
- **Llama 4 Behemoth**: ~2T parametros totales (288B activos).

### 🐳 DeepSeek
- **DeepSeek R1**: 671B parametros (MoE) centrado en razonamiento y pensamiento en cadena (Chain of Thought).
- **DeepSeek V3.2**: 685B parametros (MoE), ventana de contexto de 164K tokens.
- **DeepSeek V4 (Pro-Max)**: ~1.6T parametros totales (37B activos por token), ventana de contexto de 1 Millon de tokens.

### 🔴 Alibaba Qwen
- **Qwen 3.7-Max**: Diseñado para ejecucion de agentes autonomos de largo horizonte.
- **Qwen 3.8-Max**: Modelo multimodal masivo de **2.4 Trillones de parametros** (texto, imagen, video y documentos).

### 🟢 Google Gemma 4
- **Gemma 4 31B (Dense)**: Modelo denso optimizado para razonamiento de alto nivel con contexto de 256K.
- **Gemma 4 26B-A4B (MoE)**: ~26B parametros totales (4B activos por token).
- **Gemma 4 E-Series (E2B / E4B)**: Modelos de borde (Edge) diseñados para moviles y dispositivos locales (2B a 4B parametros).

---

## 📊 2. Tabla Comparativa: Kuro-Neko vs. Modelos 2026

| Modelo | Arquitectura | Parametros Totales | Parametros Activos / Token | Contexto Max. | Tokens Vistos en Entrenamiento | Entorno de Ejecucion |
|:---|:---|:---|:---|:---|:---|:---|
| 🐈‍⬛ **Kuro-Neko 2026** | **Dense Pura** | **7 Millones (0.007B)** | **7 Millones** | **64 – 512** | **2.5M – 50M** | **Micro-dispositivos (Raspberry Pi, relojes, terminales locales)** |
| 🟢 **Gemma 4 (E2B)** | **Dense** | **2 Mil Millones (2B)** | **2B** | **256K** | **4 – 6 Trillones** | Dispositivos moviles (Smartphones, laptops sin conexion) |
| 🦙 **Llama 4 (Scout)** | **MoE** | **109 Mil Millones (109B)** | **17B** | **10 Millones** | **15 – 20 Trillones** | Servidores de inferencia local / Nube privada |
| 🐳 **DeepSeek V4** | **MoE** | **1.6 Trillones (1,600B)** | **37B** | **1 Millon** | **25+ Trillones** | Granjas de supercomputación distribuidas |
| 🔴 **Qwen 3.8-Max** | **MoE Sparse** | **2.4 Trillones (2,400B)** | **~50B** | **1 Millon** | **30+ Trillones** | Clusters de hiper-escala en la nube |

---

## 💡 3. Conclusiones de Escala y Diseño

1. **Diferencia de Tamaño**: Kuro-Neko (7M) es aproximadamente **228,000 veces mas pequeño** en numero de parametros que un modelo de frontera como DeepSeek V4 (1.6T).
2. **Consumo de Memoria**: 
   - **DeepSeek V4**: Requiere clusters de GPUs H100/H200 acumulando Terabytes de VRAM.
   - **Kuro-Neko**: Ocupa **menos de 50 Megabytes de RAM** y corre en cualquier GPU sencilla con WGPU.
3. **Casos de Uso**: Kuro-Neko demuestra como implementar la arquitectura matematica completa de un Transformer 2026 (RoPE, RMSNorm, SwiGLU, GGUF) en codigo **Rust puro sin dependencias pesadas**.
