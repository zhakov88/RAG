# 📚 RAG – Local Retrieval-Augmented Generation System

This repository contains a **local Retrieval-Augmented Generation (RAG) pipeline** built with Python.  
It allows you to ingest documents (PDFs), create embeddings, store them in a vector store, and query them using a **local LLM via Ollama**.

✅ Fully local  
✅ No data sent to external APIs  
✅ Suitable for private / confidential documents  
✅ Includes a Panel-based web UI

Repository: https://github.com/zhakov88/RAG

---

## ✨ Features

- Local LLM inference using **Ollama**
- Modular RAG architecture
- PDF document ingestion
- Vector search for retrieval
- Interactive UI built with **Panel**
- Clean project structure, suitable for extension

---

## 🧱 Tech Stack

- **Python 3.10+**
- **uv** – fast Python package & environment manager
- **Ollama** – local LLM runtime
- **Panel** – interactive web UI
- Local embedding & vector store pipeline

---

## 🚀 Installation & Setup

### Prerequisites

- Linux / macOS / Windows  
  👉 On Windows, **WSL is strongly recommended**
- Git
- Python **3.10 or newer**

---

## 1️⃣ Install `uv`

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify:
```bash
uv --version
```

---

## 2️⃣ Install Ollama

Download from:
https://ollama.com/download

Start Ollama:
```bash
ollama serve
```

Pull model:
```bash
ollama pull llama3.1:8b
```

---

## 3️⃣ Clone Repository

```bash
git clone https://github.com/zhakov88/RAG.git
cd RAG
```

---

## 4️⃣ Create Virtual Environment

```bash
uv venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

## 5️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## 🗂️ Adding Documents

Place PDFs into:

```
data/pdf/
```

---

## ▶️ Run Application (Panel UI)

```bash
panel serve app.py --dev
```

Open:
```
http://localhost:5006/app
```

---

## 🔒 Privacy

All processing is local. No data leaves your machine.

---

## 📜 License

MIT
