# 🔍 Research and Global Landscape of LLMs in 2026 (`search.md`)

This document records the real data of the Artificial Intelligence industry in **2026** (Llama 4, DeepSeek V4, Qwen 3.8-Max, Gemma 4 families) and the direct comparison with the scale of our engine **Kuro-Neko**.

---

## 🌐 1. The Global Landscape of Models in 2026

In 2026, the industry has massively migrated toward **Mixture-of-Experts (MoE)** architectures to optimize inference speed by activating only a small part of the model for each generated token.

### 🦙 Meta Llama 4
- **Llama 4 Scout**: 109B total parameters (17B active), **10 million token** context window.
- **Llama 4 Maverick**: 400B total parameters (17B active), 1 million token context window.
- **Llama 4 Behemoth**: ~2T total parameters (288B active).

### 🐳 DeepSeek
- **DeepSeek R1**: 671B parameters (MoE) focused on reasoning and Chain of Thought.
- **DeepSeek V3.2**: 685B parameters (MoE), 164K token context window.
- **DeepSeek V4 (Pro-Max)**: ~1.6T total parameters (37B active per token), 1 million token context window.

### 🔴 Alibaba Qwen
- **Qwen 3.7-Max**: Designed for long-horizon autonomous agent execution.
- **Qwen 3.8-Max**: Massive multimodal model with **2.4 trillion parameters** (text, image, video and documents).

### 🟢 Google Gemma 4
- **Gemma 4 31B (Dense)**: Dense model optimized for high-level reasoning with a 256K context.
- **Gemma 4 26B-A4B (MoE)**: ~26B total parameters (4B active per token).
- **Gemma 4 E-Series (E2B / E4B)**: Edge models designed for mobile and local devices (2B to 4B parameters).

---

## 📊 2. Comparison Table: Kuro-Neko vs. 2026 Models

| Model | Architecture | Total Parameters | Active Parameters / Token | Max. Context | Training Tokens Seen | Execution Environment |
|:---|:---|:---|:---|:---|:---|:---|
| 🐈‍⬛ **Kuro-Neko 2026** | **Pure Dense** | **7 Million (0.007B)** | **7 Million** | **64 – 512** | **2.5M – 50M** | **Micro-devices (Raspberry Pi, watches, local terminals)** |
| 🟢 **Gemma 4 (E2B)** | **Dense** | **2 Billion (2B)** | **2B** | **256K** | **4 – 6 Trillion** | Mobile devices (smartphones, offline laptops) |
| 🦙 **Llama 4 (Scout)** | **MoE** | **109 Billion (109B)** | **17B** | **10 Million** | **15 – 20 Trillion** | Local inference servers / Private cloud |
| 🐳 **DeepSeek V4** | **MoE** | **1.6 Trillion (1,600B)** | **37B** | **1 Million** | **25+ Trillion** | Distributed supercomputing farms |
| 🔴 **Qwen 3.8-Max** | **MoE Sparse** | **2.4 Trillion (2,400B)** | **~50B** | **1 Million** | **30+ Trillion** | Hyperscale cloud clusters |

---

## 💡 3. Scale and Design Conclusions

1. **Size Difference**: Kuro-Neko (7M) is approximately **228,000 times smaller** in number of parameters than a frontier model like DeepSeek V4 (1.6T).
2. **Memory Consumption**: 
   - **DeepSeek V4**: Requires clusters of H100/H200 GPUs accumulating terabytes of VRAM.
   - **Kuro-Neko**: Uses **less than 50 Megabytes of RAM** and runs on any simple GPU with WGPU.
3. **Use Cases**: Kuro-Neko demonstrates how to implement the complete mathematical architecture of a 2026 Transformer (RoPE, RMSNorm, SwiGLU, GGUF) in **pure Rust code without heavy dependencies**.
