# multi-modal-rag

## Overview
This project implements a **strict Retrieval-Augmented Generation (RAG) system** that answers questions **only using the contents of a provided PDF document**.

The system avoids hallucinations by explicitly refusing to answer **queries not supported by the document**, ensuring faithful, document-grounded responses.

---

## Features
- Multi-modal ingestion: **text, tables, OCR** from PDF documents  
- **FAISS-based vector retrieval** with persistent local index  
- **Document-grounded question answering** with source citations  
- Explicit refusal for unsupported or out-of-scope queries  
- CLI and Streamlit-based **interactive interfaces**

---

## Setup

1. **Clone the repository**

2. **Create a virtual environment** :  

```bash
python -m venv .venv
.venv\Scripts\activate 
```
3. **Install dependencies:** : 
```bash
pip install -r requirements.txt

```
4. **Set API keys in a .env file** : 

OPENAI_API_KEY=your_api_key_here

## Running the Application

**CLI Mode (Terminal)** :

```bash
python app.py
```

**UI MODE (Stremlit)**

```bash
streamlit run streamlit_app.py
```