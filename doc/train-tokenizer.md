# 🔤 Kuro-Neko — Entrenamiento del Tokenizador (`train-tokenizer`)

El subcomando `train-tokenizer` permite entrenar un tokenizador **BPE a Nivel de Bytes (Byte-Level Byte-Pair Encoding)** adaptado al vocabulario y dominio específico de tu conjunto de datos de entrenamiento.

Al finalizar el entrenamiento, el subcomando guarda el modelo de tokenización en el archivo `custom_tokenizer.json`, el cual es consumido automáticamente por los subcomandos `train`, `generate` y `export`.

---

## ⚙️ Tabla Completa de Parámetros CLI

| Flag | Tipo | Valor por Defecto | Descripción Detallada |
| :--- | :--- | :--- | :--- |
| `--dataset <repo>` | `Vec<String>` (separados por coma) | `[]` | Identificador(es) del repositorio de Hugging Face (ej. `roneneldan/TinyStories`). Múltiples repositorios pueden combinarse para construir el corpus. |
| `--file <path>` | `Option<String>` | `None` | Ruta a un archivo de texto local en el sistema para utilizarlo como fuente del corpus de entrenamiento del tokenizador. |
| `--dataset-file <filename>` | `String` | `TinyStories-train.txt` | Nombre del archivo objetivo a descargar del repositorio de Hugging Face. |
| `--vocab-size <n>` | `usize` | `8000` | Tamaño total del vocabulario a generar (número total de subpalabras y tokens aprendidos). |

---

## 🧩 ¿Qué es la Tokenización BPE Byte-Level?

El algoritmo **Byte-Level Byte-Pair Encoding (BPE)** opera directamente sobre los bytes UTF-8 sin procesar en lugar de subpalabrasUnicode explícitas:

1. **Representación Base de 256 Bytes**: Inicialmente, los 256 bytes individuales (`0x00` a `0xFF`) forman la base del vocabulario.
2. **Fusión Iterativa de Pares Frecuentes**: El algoritmo analiza el corpus de texto y encuentra repetidamente el par de bytes o tokens contiguos más frecuente, fusionándolo en un nuevo token único.
3. **Cero Tokens Desconocidos (`<unk>`)**: Dado que cualquier carácter o secuencia Unicode en cualquier idioma (incluyendo emojis, símbolos matemáticos o código binario) puede descomponerse en sus bytes UTF-8 constitutivos, un tokenizador Byte-Level **nunca falla ni genera tokens `<unk>`**.

---

## 🔠 Normalización Unicode NFKC

Antes de proceder con el cálculo de frecuencias BPE, Kuro-Neko aplica un pipeline de normalización Unicode estricto utilizando el estándar **NFKC (Normalization Form KC - Compatibility Decomposition followed by Canonical Composition)**:

- **Unificación de Caracteres**: Convierte caracteres visualmente similares o variantes de compatibilidad (como ligaduras, superíndices, variantes de ancho completo) a una representación canónica normalizada.
- **Consistencia en el Modelo**: Asegura que variantes del mismo carácter no consuman múltiples entradas en la tabla de vocabulario de forma redundante.

---

## 🏷️ Tokens Especiales

El tokenizador entrenado incluye y reserva dos tokens especiales fundamentales para la gestión de secuencias en modelos de lenguaje:

| Token Especial | Función |
| :--- | :--- |
| `<|endoftext|>` | Indica la frontera de fin de documento o de secuencia. Utilizado durante el entrenamiento para separar historias o documentos independientes y durante la inferencia para detener la generación autorregresiva. |
| `<|padding|>` | Token utilizado para rellenar (*pad*) lotes de secuencias de longitud impar cuando se procesan en la GPU. |

---

## 💾 Archivo de Salida: `custom_tokenizer.json`

Al completar el proceso, Kuro-Neko serializa la estructura completa del tokenizador en formato JSON en el archivo `custom_tokenizer.json` en la raíz del proyecto. Este archivo contiene:

- El mapa de fusiones BPE (*merges*).
- La tabla de vocabulario token-a-ID e ID-a-token.
- Las reglas de pre-tokenización (división por espacios en blanco y puntuación).
- La configuración de normalización NFKC y de tokens especiales.

---

## 💻 Ejemplos de Uso

### Ejemplo 1: Entrenar Tokenizador desde un Repositorio de Hugging Face
Entrena un tokenizador de 8,000 tokens sobre el dataset `roneneldan/TinyStories`:
```bash
cargo run --release -- train-tokenizer --dataset roneneldan/TinyStories --vocab-size 8000
```

### Ejemplo 2: Entrenar Tokenizador sobre un Archivo Local
Entrena un tokenizador más compacto (vocabulario de 4,096 tokens) a partir de un archivo local:
```bash
cargo run --release -- train-tokenizer --file ./mi_corpus_local.txt --vocab-size 4096
```

### Ejemplo 3: Entrenar Tokenizador Gran Vocabulario para Corpus Combinado
Entrena un tokenizador amplio (vocabulario de 16,384 tokens) combinando múltiples datasets:
```bash
cargo run --release -- train-tokenizer \
  --dataset roneneldan/TinyStories,wikitext \
  --vocab-size 16384
```
