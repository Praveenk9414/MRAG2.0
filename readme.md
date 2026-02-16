# 🚀 Multimodal RAG System (SIH Ready)

A full-stack **Multimodal Retrieval-Augmented Generation (RAG)** application that enables users to ingest and query **PDFs, DOCX, Images, and Audio** using natural language.

The system builds a **unified semantic search layer** across modalities and generates grounded answers with source transparency.

---
## 🎥 Project Demo

Watch the working demo of the Multimodal RAG system:

👉 https://www.youtube.com/watch?v=oPnFlJ6WeQs

---
## ✨ Features

### 📥 Multimodal Ingestion

* ✅ PDF text extraction
* ✅ DOCX text extraction
* ✅ Image embeddings (CLIP)
* ✅ Audio → Speech-to-Text (Whisper)
* ✅ Automatic chunking and indexing

### 🔎 Semantic & Cross-Modal Search

* Natural language querying
* Unified vector space for all modalities
* Re-ranking using cross-encoder
* Supports:

  * Text → Text/Image/Audio retrieval
  * Image → related documents
  * Audio → related content

### 🧠 Grounded Answer Generation

* LLM-generated responses
* Context-aware summaries
* Source-grounded outputs
* Reduced hallucinations

### 📊 Citation Transparency

* Numbered citations
* Expand to view source chunks
* Inspect metadata

---

## 🏗️ Project Structure

```
llm-rag-with-reranker-demo/
├── app.py / multimodal_rag.py   # Main Streamlit app
├── models/
│   └── whisper-small.en/        # Local Whisper model
├── demo-rag-chroma/             # Vector DB persistence
├── requirements.txt
├── Makefile
└── README.md
```

---

## ⚙️ Tech Stack

**LLM & Embeddings**

* Ollama (llama3.2)
* nomic-embed-text
* Cross-Encoder (MS MARCO)

**Multimodal**

* OpenCLIP / SentenceTransformers
* OpenAI Whisper (offline)

**Backend**

* Python
* LangChain
* ChromaDB

**Frontend**

* Streamlit

---

## 🧩 Required Models

### 🔹 Ollama Models

Install:

```bash
ollama pull nomic-embed-text:latest
ollama pull llama3.2:3b
```

Recommended:

* `llama3.2:3b` → lightweight
* `llama3.2:7b` → better quality (if RAM allows)

---

### 🔹 Whisper Model (Audio)

We use:

```
openai/whisper-small.en
```

It will be downloaded automatically to:

```
./models/whisper-small.en
```

---

## 🔧 Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yankeexe/llm-rag-with-reranker-demo.git
cd llm-rag-with-reranker-demo
```

---

### 2️⃣ Create environment

```bash
conda create -n rag python=3.10
conda activate rag
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install \
streamlit \
chromadb \
langchain \
sentence-transformers \
PyMuPDF \
python-docx \
Pillow \
openai-whisper \
transformers \
torchaudio
```

---

### 4️⃣ Start Ollama

```bash
ollama serve
```

---

## ▶️ Running the App

### Option A — Direct

```bash
streamlit run multimodal_rag.py
```

### Option B — Using Makefile

```bash
make run
```

App will open at:

```
http://localhost:8501
```

---

## 📌 Usage

### Upload Data

Supported formats:

* 📄 PDF
* 📝 DOCX
* 🖼️ PNG/JPG
* 🎧 MP3/WAV

Click **⚡ Process** to index.

---

### Ask Questions

Examples:

* “Summarize the international development report.”
* “Show the email screenshot.”
* “What was discussed in the meeting audio?”
* “Find the document related to the 14:32 screenshot.”

---

## 🧠 How It Works

```
Upload → Parse → Chunk → Embed → Store (ChromaDB)
                                    ↓
User Query → Embed → Retrieve → Re-rank → LLM → Answer
```

---

## 🚧 Future Improvements

* [ ] True cross-modal CLIP alignment
* [ ] Video support
* [ ] Better citation UI
* [ ] Streaming audio search
* [ ] Agentic retrieval workflows

---

## 🤝 Acknowledgements

* Ollama
* LangChain
* ChromaDB
* OpenAI Whisper
* Sentence Transformers
* Original repo: yankeexe

---

## 🏆 SIH Readiness

This project satisfies:

✅ Multimodal ingestion
✅ Unified vector search
✅ Grounded LLM answers
✅ Citation transparency
✅ Cross-format linking

---

**Built with focus on real-world multimodal intelligence.**
