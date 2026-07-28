# 📱 Post para LinkedIn (`post.md`)

*Copia y pega este texto directamente en LinkedIn:*

---

¿Por qué usar PyTorch y Python cuando puedes construir tu propio motor de LLM desde cero en Rust?

Llevo un tiempo desarrollando Kuro-Neko, un motor de entrenamiento e inferencia de Modelos de Lenguaje creado en Rust puro, acelerado por GPU con Burn 0.21 y WGPU, capaz de exportar directamente a paquetes GGUF v3.

Nada de llamar a la API de OpenAI ni de hacer wrappers. Es la implementación matemática completa de la arquitectura Transformer de 2026, escrita línea a línea:

Arquitectura de Última Generación:
- RoPE (Rotary Position Embeddings) para una memoria posicional dinámica y sin tablas fijas.
- RMSNorm para normalización ultrarrápida sin sesgo.
- SwiGLU FFN para la capa de avance gating con activación SiLU.
- Muestreo moderno: Top-K, Top-P Nucleus y Repetition Penalty.

Los retos reales de ingeniería de sistemas:
- Tokenización en streaming sin copia de memoria: De 11 a 40 MB/s procesados en paralelo, manteniendo el consumo por debajo de 2.4 GB de RAM.
- Optimización salvaje de VRAM: De gastar 262 MB por batch pasamos a solo 32 KB por paso (8.000 veces menos asignación en memoria) al sustituir matrices one-hot densas por cálculo esparso de pérdida NLL.
- Cero cuellos de botella PCIe: Precálculo de frecuencias RoPE directamente en la VRAM de la tarjeta gráfica.

Exportación nativa a GGUF v3:
Programé un serializador binario nativo que empaqueta los tensores alineados a 32 bytes con la cabecera LLaMA y el diccionario BPE embebido. Para resolver el choque de diseño entre la rotación RoPE por mitades (estilo Hugging Face/Python) y la rotación intercalada (estilo llama.cpp/C++), implementé la función de permutación de pesos oficial en Rust.

¿El resultado? El modelo se ejecuta de forma nativa en llama.cpp procesando prompts a 2.600 tokens/s y generando texto a 200 tokens/s, además de abrirse directamente en LM Studio u Ollama sin configurar nada.

Picar un LLM "desde el metal" te enseña sobre atención, memoria de GPU e ingeniería de sistemas más que años consumiendo APIs de terceros.

¡Todo el código y la documentación están disponibles en el repositorio!

#Rust #ArtificialIntelligence #MachineLearning #DeepLearning #LLM #GGUF #LlamaCPP #SystemsEngineering #HuggingFace
