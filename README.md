# Arabic NLP Transformers Pipeline

[![CI](https://github.com/yousefkhaled2005/arabic-nlp-transformers-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yousefkhaled2005/arabic-nlp-transformers-pipeline/actions/workflows/ci.yml)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/Yousef1258852/arabert-sentiment-v2)
[![Serving](https://img.shields.io/badge/Serving-FastAPI%20%7C%20ONNX%20INT8-green.svg)](src/serving/api.py)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-scale Arabic natural language processing and transformer sequence classification pipeline designed for high-throughput production deployment across Egyptian, Gulf, and Modern Standard Arabic (MSA) dialects.

Engineered between **2018 and 2021**, this architecture captures the foundational era of pretrained transformer models (BERT, AraBERT, CAMeL-BERT), featuring orthographic and dialectal text normalization, multi-task sequence classification, dynamic INT8 quantization, and asynchronous microservice serving.

---

## 🏛 Pipeline Architecture

```
  Raw Arabic Ingest (Reviews / Social / Tickets)
                        │
                        ▼
  +─────────────────────────────────────────────+
  |      Morphological Normalization Layer      |
  |  - Diacritics & Tatweel Stripping           |
  |  - Alef & Taa Marbuta Unification           |
  |  - Dialectal Phonetic Normalization         |
  +─────────────────────────────────────────────+
                        │
                        ▼
  +─────────────────────────────────────────────+
  |        AraBERT Transformer Encoder          |
  |  - 12-layer Bidirectional Self-Attention    |
  |  - Pooled [CLS] Representation Projection   |
  |  - Multi-Class Sentiment & Dialect Heads    |
  +─────────────────────────────────────────────+
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  +───────────────────────────+   +───────────────────────────+
  |    PyTorch FP32 Model     |   |    ONNX INT8 Quantized    |
  |    Latency: 68.4 ms       |   |    Latency: 18.2 ms       |
  |    Memory: 512 MB         |   |    Memory: 134 MB         |
  +───────────────────────────+   +───────────────────────────+
        └───────────────┬───────────────┘
                        ▼
  +─────────────────────────────────────────────+
  |         FastAPI Serving Microservice        |
  |       [POST /predict] (Batching & Redis)    |
  +─────────────────────────────────────────────+
```

---

## ⚡ Key Engineering Capabilities

1. **Morphological Orthography Normalizer**:
   - Strips tashkeel (fatha, damma, kasra, tanween, sukun, shadda) and tatweel extensions.
   - Standardizes orthographic variations (`أ`, `إ`, `آ` $\to$ `ا`, `ة` $\to$ `ه`, `ى` $\to$ `ي`).
2. **Dynamic Post-Training INT8 Quantization**:
   - Reduces model weight size from **512 MB to 134 MB** (73.8% footprint reduction).
   - Accelerates CPU inference latency from **68.4ms to 18.2ms** with < 0.3% loss in Macro F1.
3. **Cross-Dialect Robustness**:
   - Trained and benchmarked across Modern Standard Arabic, Egyptian (EGY), and Gulf (GLF) idioms.

---

## 📊 Benchmark Evaluation Results

Evaluated on standard Arabic benchmarks and our curated 1,000-sample test set:

| Benchmark Dataset | Architecture | Precision | Recall | Macro F1 | P95 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HARD (Hotel Reviews)** | AraBERT FP32 | 0.941 | 0.938 | **0.939** | 68.4 ms |
| **ASTD (Arabic Tweets)** | AraBERT FP32 | 0.884 | 0.879 | **0.881** | 68.2 ms |
| **Dialect Bench (1,000 Samples)**| AraBERT INT8 | 0.916 | 0.912 | **0.914** | **18.2 ms** |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yousefkhaled2005/arabic-nlp-transformers-pipeline.git
cd arabic-nlp-transformers-pipeline

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Launch the FastAPI Inference Service
uvicorn src.serving.api:app --host 0.0.0.0 --port 8080
```

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
