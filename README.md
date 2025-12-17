# Hybrid RAG System (OpenAI + Ollama)

This project is a robust Retrieval-Augmented Generation (RAG) system that supports both cloud-based (OpenAI) and local (Ollama) AI models. It ingests PDF documents, extracts text and structure using **Docling**, and enables Q&A based on the document content.

## Features

- **Hybrid AI Support**: Switch seamlessly between OpenAI (`gpt-4o-mini`) and Local Ollama (`llama3.1`) models.
- **Advanced PDF Processing**: Uses **Docling** for high-quality OCR and layout analysis.
- **Structured Data Extraction**: Converts unstructured PDF text into structured JSON format.
- **Vector Search**: Uses **Supabase (pgvector)** for storing and retrieving document embeddings.
- **Interactive UI**: Built with **Streamlit** for easy document upload and chatting.

## Project Structure

```bash
CUSTOM_RAG/
│── app.py                 # Streamlit UI (Frontend)
│── pdf_utils.py           # PDF processing with Docling & JSON structuring
│── rag_backend.py         # RAG logic, Embeddings, DB connection
│── .env                   # API keys & Config (not versioned)
│── requirements.txt       # Python Dependencies
└── README.md              # Documentation
```

## Technologies Used

- **Python 3.11+**
- **Streamlit** (Frontend)
- **Docling** (PDF/OCR Processing)
- **Supabase / pgvector** (Vector Database)
- **OpenAI API** (Cloud LLM & Embeddings)
- **Ollama** (Local LLM & Embeddings)

## Installation

### 1. Clone the repository

```bash
git clone <REPO_URL>
cd CUSTOM_RAG
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & Configure Ollama (For Local Mode)

1. Download and install [Ollama](https://ollama.com/).
2. Pull the required models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Environment Setup

Create a `.env` file inside the project folder:

```bash
OPENAI_API_KEY=your_openai_api_key
SUPABASE_DB_URL=postgresql://postgres:password@db.project.supabase.co:5432/postgres
```

## Database Setup (Supabase)

1. **Create a Supabase Project**: Go to [database.new](https://database.new).
2. **Enable Vector Extension**: Run this in the Supabase SQL Editor:

```sql
create extension if not exists vecs;
create schema if not exists vecs;
```

3. **Create Collections**:
   The application automatically manages collections via the `vecs` python client, but you can manually verify them.
   - `rag_docs`: For OpenAI embeddings (1536 dimensions).
   - `rag_docs_local`: For Ollama embeddings (768 dimensions).

## Run the Application

```bash
streamlit run app.py
```

After starting, open your browser at: `http://localhost:8501`

## Usage

1. **Select Provider**: Choose "OpenAI" or "Local (Ollama)" in the sidebar.
2. **Upload PDF**: Upload a document to ingest.
3. **Ask Questions**: Chat with your document in the main window.

- Exporting structured document JSON