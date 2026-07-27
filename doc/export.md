# 📦 Kuro-Neko — Exportación GGUF v3 (`export`)

El subcomando `export` convierte los pesos del modelo guardados en el directorio `model_output/` junto con la información del tokenizador BPE a un archivo binario único en formato **GGUF v3**.

GGUF (GGML Unified Format) es el estándar universal adoptado por la comunidad para empaquetar modelos de lenguaje e hiperparámetros de arquitectura en un solo archivo binario de alto rendimiento, optimizado para inferencia en CPU y GPU.

---

## ⚙️ Tabla Completa de Parámetros CLI

| Flag | Tipo | Valor por Defecto | Descripción Detallada |
| :--- | :--- | :--- | :--- |
| `--model-name <name>` | `String` | `kuro-neko-qwen` | Nombre base del modelo binario exportado. Define el nombre del archivo resultante dentro del directorio de salida. |

---

## 🛠️ El Formato Binario GGUF v3

El formato **GGUF v3** está diseñado para permitir una lectura secuencial rápida y asignación de memoria mapeada (`mmap`):

1. **Cabecera Magic de 4 Bytes**: Contiene el número mágico `0x46554747` (`GGUF` en ASCII).
2. **Versión de Formato**: Entero de 32 bits con valor `3`.
3. **Contador de Tensores y Pares KV**: Declara cuántos tensores de pesos y cuántos pares clave-valor de metadatos contiene el archivo.
4. **Metadatos KV Estruturados**: Claves tipo cadena con valores tipados (enteros, flotantes, arreglos de cadenas).
5. **Alineación Directa de Tensores**: La sección de datos binarios de los tensores comienza inmediatamente después de la cabecera de metadatos, alineada a bloques estrictos.

---

## 📐 Alineación de Tensores a 32 Bytes (`general.alignment`)

Kuro-Neko escribe la clave de metadatos especial:
$$\text{general.alignment} = 32$$

- **Beneficio**: Garantiza que el desplazamiento (*offset*) de byte de inicio de cada tensor dentro del binario sea un múltiplo exacto de 32.
- **Rendimiento GPU**: Esta alineación es un requisito indispensable para la aceleración SIMD en CPU (AVX-512, ARM Neon) y para la carga directa en memoria GPU sin re-alineación previa en memoria principal.

---

## 🏛️ Metadatos de Arquitectura Mapeados

Durante la exportación, Kuro-Neko mapea la configuración interna `ModelConfig` a las claves universales de la arquitectura `"llama"` en GGUF para garantizar máxima compatibilidad con runtime externos:

| Clave GGUF KV | Valor / Origen en Kuro-Neko | Descripción |
| :--- | :--- | :--- |
| `general.architecture` | `"llama"` | Tipo de arquitectura base reconocida universalmente. |
| `general.name` | `<model-name>` | Nombre del modelo exportado. |
| `general.alignment` | `32` | Alineación de memoria en bytes. |
| `llama.context_length` | `model_config.n_ctx` | Longitud máxima de ventana de contexto. |
| `llama.embedding_length` | `model_config.n_embd` | Dimensión de embedding $D$. |
| `llama.block_count` | `model_config.n_layer` | Número de capas/bloques Transformer. |
| `llama.attention.head_count` | `model_config.n_head` | Número de cabezas de atención Query. |
| `llama.attention.head_count_kv` | `model_config.n_kv_head` | Número de cabezas KV para GQA. |
| `llama.rope.freq_base` | `10000.0` | Base de frecuencia RoPE ($\theta$). |
| `tokenizer.ggml.model` | `"gpt2"` / `"bpe"` | Tipo de modelo de tokenización. |
| `tokenizer.ggml.tokens` | `Vec<String>` | Lista de todos los tokens del vocabulario. |
| `tokenizer.ggml.scores` | `Vec<f32>` | Puntuaciones o log-probabilidades BPE. |
| `tokenizer.ggml.token_type` | `Vec<i32>` | Tipo de token (Normal, Control, Desconocido). |

---

## 🎯 Ubicación del Archivo Exportado

El binario exportado se guarda automáticamente en la ruta:
```text
exported/<model-name>/model.gguf
```
*(O `exported/<model-name>.gguf` según la versión de ejecución).*

---

## 🧩 Compatibilidad de Ecosistema

Los binarios `.gguf` generados por Kuro-Neko son 100% compatibles con los motores de inferencia más populares del mercado:

- 🦙 **llama.cpp**: Permite ejecutar inferencia en terminal o servidor HTTP.
- 💻 **LM Studio**: Permite cargar el modelo mediante la opción *Import local model* y chatear con él de forma visual.
- 🦙 **Ollama**: Permite crear un `Modelfile` apuntando al binario exportado y desplegarlo como microservicio local.

---

## 💻 Ejemplos de Exportación y Despliegue

### 1. Exportar el Modelo
```bash
cargo run --release -- export --model-name kuro-neko-v1
```

### 2. Ejecutar con llama.cpp
```bash
./llama-cli -m exported/kuro-neko-v1/model.gguf -p "Once upon a time" -n 100
```

### 3. Crear un Modelo en Ollama
Crea un archivo llamado `Modelfile`:
```dockerfile
FROM ./exported/kuro-neko-v1/model.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```
Luego ejecuta:
```bash
ollama create kuro-neko -f Modelfile
ollama run kuro-neko "Once upon a time"
```
