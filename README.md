## RAG System with OCR Support

This project is a simple Retrieval-Augmented Generation (RAG) system that ingests documents and answers questions based on their content.
PDF documents are processed, text is extracted (OCR is used when text cannot be read), and chunks are stored in a vector database.

## Project Structure

```bash
CUSTOM_RAG/
│── app.py                 # Streamlit UI
│── pdf_utils.py           # Text extraction + OCR + structuring
│── rag_backend.py         # Chunks, embeddings, querying, RAG logic
│── .env                   # API keys (not versioned)
│── requirements.txt       # Dependencies
└── README.md
```

## Technologies Used
- Python
- Streamlit
- OpenAI API
- Supabase Vector Store
- PyPDF2
- OCR Engine (EasyOCR or PaddleOCR)
- vecs
- NumPy

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

- Note:
On macOS, PaddleOCR may crash due to native dependencies.
If so, use EasyOCR as an alternative.

## Environment Setup

Create a `.env` file inside the project folder:

```bash
OPENAI_API_KEY=your_api_key_here
SUPABASE_DB_URL=your_supabase_vector_url
```

## Run the Application

```bash
streamlit run app.py
```

After starting, open:
- http://localhost:8501

## Possible Improvements

- Page-level metadata storage
- Support ingestion of multiple PDFs grouped by source
- Preview of extracted text
- Parallel OCR processing
- Exporting structured document JSON