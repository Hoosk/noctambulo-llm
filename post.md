# 📱 LinkedIn Post (`post.md`)

*Copy and paste this text directly into LinkedIn:*

---

Why use PyTorch and Python when you can build your own LLM engine from scratch in Rust?

I have been developing Noctámbulo for a while, a Language Model training and inference engine built in pure Rust, GPU-accelerated with Burn 0.22 and WGPU, capable of exporting directly to GGUF v3 packages.

No calling the OpenAI API and no wrappers. It is the complete mathematical implementation of the 2026 Transformer architecture, written line by line:

State-of-the-Art Architecture:
- RoPE (Rotary Position Embeddings) for dynamic positional memory without fixed tables.
- RMSNorm for ultra-fast normalization without bias.
- SwiGLU FFN for the gated feed-forward layer with SiLU activation.
- Modern sampling: Top-K, Top-P Nucleus and Repetition Penalty.

The real systems engineering challenges:
- Streaming tokenization without memory copies: From 11 to 40 MB/s processed in parallel, keeping consumption below 2.4 GB of RAM.
- Wild VRAM optimization: From spending 262 MB per batch we went to only 32 KB per step (8,000 times less memory allocation) by replacing dense one-hot matrices with sparse NLL loss computation.
- Zero PCIe bottlenecks: Precomputation of RoPE frequencies directly in the graphics card's VRAM.

Native export to GGUF v3:
I programmed a native binary serializer that packs the tensors aligned to 32 bytes with the LLaMA header and the embedded BPE dictionary. To resolve the design clash between half-split RoPE rotation (Hugging Face/Python style) and interleaved rotation (llama.cpp/C++ style), I implemented the official weight permutation function in Rust.

The result? The model runs natively in llama.cpp processing prompts at 2,600 tokens/s and generating text at 200 tokens/s, besides opening directly in LM Studio or Ollama without configuring anything.

Hacking an LLM "from the metal" teaches you about attention, GPU memory and systems engineering more than years of consuming third-party APIs.

All the code and documentation are available in the repository!

#Rust #ArtificialIntelligence #MachineLearning #DeepLearning #LLM #GGUF #LlamaCPP #SystemsEngineering #HuggingFace
