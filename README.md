# llama3_RAG_test_v1
# LLaMA 3 RAG Chatbot

This repository contains a **Retrieval-Augmented Generation (RAG) chatbot** built with **LLaMA 3‑8B** and **FAISS**, designed to answer questions from your personal documents (like PDFs). It uses **Ollama** for CPU-friendly LLaMA inference and **LangChain** for vector retrieval and embeddings.

---

## 🚀 Features

- Query personal PDFs or documents in natural language.
- FAISS-based semantic search for efficient retrieval.
- CPU-friendly setup (16 GB RAM, 12 logical CPU cores tested).
- Includes **thinking dots** to visualize when the model is processing.
- Handles all dependency issues and LangChain deprecations.
- Modular Python scripts:
  - `ingest.py` — index your PDFs
  - `rag_chain.py` — loads LLaMA + FAISS retriever
  - `chat.py` — interactive chatbot

---

## 📁 Repository Structure
basic_AI_chatbot/
│
├── data/
│ └── documents/
│ └── CV-Ali Shahid.pdf
├── vectorstore/
├── src/
│ ├── ingest.py
│ ├── rag_chain.py
│ └── chat.py
├── requirements.txt
└── README.md


> **Note:** `vectorstore/` and PDFs are ignored in Git (`.gitignore`) for privacy and size.

---

## 🛠 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/alishahidklair/llama3_RAG_test_v1.git
cd llama3_RAG_test_v1

### 2️⃣ Create a Python virtual environment

python3 -m venv AI_chatbot
source AI_chatbot/bin/activate

### 3️⃣ Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

Dependencies: LangChain, LangChain Ollama & HuggingFace modules, FAISS CPU, PyPDF, Sentence Transformers, Ollama client, etc.

### 4️⃣ Install and run Ollama & pull LLaMA 3‑8B



1. Install Ollama (see Ollama official docs: https://ollama.com/)

2. Pull the model:

ollama pull llama3


3. Start Ollama server:

ollama serve


The server must be running whenever you use the chatbot.


📄 Index Your Documents

Place your PDF documents in:

data/documents/


Then run:

python src/ingest.py


This creates a FAISS index in vectorstore/ for fast semantic search.

You only need to do this once unless you add new documents.

💬 Run the Chatbot
python src/chat.py


Ask questions interactively.

The model uses a thinking dots indicator so you know it’s processing.

Example:

Ask something (or 'exit'): What skills are mentioned in the CV?
...
Answer:
Python, JavaScript, SQL, Data Analysis


Type exit to quit.

CPU Note: On 16 GB RAM and 12 logical cores, replies may take 2–5 minutes per query. Consider smaller models for faster testing.

⚡ Common Issues & Fixes

LangChain deprecation warnings

The class `Ollama` or `HuggingFaceEmbeddings` is deprecated.


Fixed by installing new packages:

pip install -U langchain-ollama langchain-huggingface


Import from updated modules:

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


Module not found errors

Make sure you are in the virtual environment and installed dependencies:

source AI_chatbot/bin/activate
pip install -r requirements.txt


Git conflicts while pushing to GitHub

Resolve merge conflicts, then:

git add <file>
git rebase --continue
git push -u origin main


Or force push if the remote is empty:

git push -u origin main --force


FAISS / vectorstore issues

Make sure to run ingest.py before chat.py.

Never push vectorstore/ to GitHub (large files, private data).

⚙️ Performance Tips

Set CPU threads in rag_chain.py:

import os
os.environ["OMP_NUM_THREADS"] = "12"


Use int8 quantization for LLaMA 3‑8B (handled automatically by Ollama).

Keep the model loaded between queries to avoid long reload times.

📌 References

Ollama
 — Model server and LLaMA 3‑8B CPU inference

LangChain
 — RAG pipelines

FAISS
 — Vector search

Sentence Transformers
 — Embeddings for semantic search

📝 License

This project is licensed under the MIT License — see LICENSE for details.
You are free to use, modify, and distribute this code, with attribution.


---

If you want, I can also **write the matching `LICENSE` MIT file** with your name and year so it’s ready to push to GitHub.  

Do you want me to do that next?












