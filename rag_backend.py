from dotenv import load_dotenv
import os
import vecs
from openai import OpenAI
from typing import List

# Variablen aus .env-Datei laden:
load_dotenv()

# Wichtige Konfigurationswerte:
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Fehler ausgeben, falls wichtige Werte fehlen:
if not SUPABASE_DB_URL:
    raise ValueError("SUPABASE_DB_URL ist nicht gesetzt (.env-Datei prüfen)!")

# Configs für die verschiedenen KI-Provider
PROVIDERS = {
    "OpenAI": {
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o-mini",
        "collection_name": "rag_docs",
        "dimension": 1536,
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "Local (Ollama)": {
        "embedding_model": "nomic-embed-text",
        "chat_model": "llama3.1",
        "collection_name": "rag_docs_local",
        "dimension": 768,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}


def get_config(provider):
    return PROVIDERS.get(provider, PROVIDERS["Local (Ollama)"])


def get_client(provider):
    config = get_config(provider)
    if provider == "OpenAI":
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"{config['api_key_env']} not set")
        return OpenAI(api_key=api_key)
    else:
        return OpenAI(base_url=config["base_url"], api_key=config["api_key"])


def get_collection(provider):
    config = get_config(provider)
    return vx.get_or_create_collection(
        name=config["collection_name"], dimension=config["dimension"]
    )


# Einstiegspunkt für die Verbindung zur Supabase-Datenbank:
# Wir erzwingen SSL für eine sichere Verbindung
if "?" not in SUPABASE_DB_URL:
    SUPABASE_DB_URL += "?sslmode=require"
elif "sslmode" not in SUPABASE_DB_URL:
    SUPABASE_DB_URL += "&sslmode=require"

vx = vecs.create_client(SUPABASE_DB_URL)

# Helper
def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Teilt einen langen Text in kleinere Stücke (Chunks).
    """
    chunks = []
    current = []
    current_len = 0

    # Wir trennen grob nach Zeilenumbrüchen:
    for symbol in text.split("\n"):
        # Wenn der aktuelle Chunk zu groß werden würde, speichern wir ihn ab:
        if current_len + len(symbol) > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(symbol)
        current_len += len(symbol)

    # Restlichen Text, falls vorhanden, noch als letzten Chunk hinzufügen:
    if current:
        chunks.append("\n".join(current))

    return chunks

# Helper
def embed_texts(texts: List[str], provider: str) -> List[List[float]]:
    """
    Erzeugt Embeddings für eine Liste von Texten.
    """
    client = get_client(provider)
    config = get_config(provider)

    response = client.embeddings.create(input=texts, model=config["embedding_model"])

    # "response.data" ist eine Liste von Objekten, die jeweils ein Embedding enthalten:
    return [data.embedding for data in response.data]


def ingest_document(raw_text: str, source: str, provider: str) -> int:
    """
    Nimmt einen Rohtext, zerteilt ihn in Chunks, erstellt Embeddings
    und speichert alles in der Supabase-Collection.
    """
    # 1. Text in Chunks aufteilen:
    chunks = chunk_text(text=raw_text)
    if not chunks:
        return 0
    print(f"Anzahl Chunks: {len(chunks)}")

    # 2. Embeddings für alle Chunks erzeugen:
    embeddings = embed_texts(texts=chunks, provider=provider)

    # 3. Items für "vecs" vorbereiten:
    items = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        item_id = f"{source}_{i}"
        metadata = {"source": source, "chunk": i, "text": chunk}
        items.append((item_id, emb, metadata))

    print(f"Anzahl der Items für 'upsert': {len(items)}")

    # 4. Alle Items in die Collection schreiben:
    collection = get_collection(provider)
    collection.upsert(items)

    # 5. Index für schnellere Ähnlichkeitssuche erstellen:
    collection.create_index()

    print(f"{len(items)} Chunks von '{source}' gespeichert.")
    return len(items)


def embed_query(query: str, provider: str) -> List[float]:
    """
    Erzeugt ein einzelnes Embedding für eine Nutzerfrage (Query).
    """
    client = get_client(provider)
    config = get_config(provider)

    resp = client.embeddings.create(model=config["embedding_model"], input=[query])

    return resp.data[0].embedding

# Helper
def search_similar_chunks(query: str, provider: str, k: int = 10):
    """
    Sucht die k ähnlichsten Chunks in der Collection für eine
    gegebene Nutzerfrage (query).
    """
    query_vec = embed_query(query=query, provider=provider)

    collection = get_collection(provider)

    result = collection.query(
        data=query_vec,
        limit=k,
        measure="cosine_distance",
        include_metadata=True,
        include_value=True,
    )

    return result

# Helper
def build_rag_prompt(question: str, results: List[tuple]) -> str:
    """
    Erzeugt einen vollständigen Prompt für ein RAG-Chatmodell.
    """
    kontexte = []
    for vec_id, score, metadata in results:
        kontexte.append(metadata["text"])

    kontext_block = "\n\n---\n\n".join(kontexte)

    prompt = f"""
        ## Allgemein:
        Du bist ein hilfreicher Assistent. Beantworte die Frage ausschließlich mit Hilfe
        des bereitgestellten Kontextes. Wenn die Frage im Kontext nicht klar steht, sage ehrlich, dass du es nicht weißt.
        
        ## Frage:
        {question}
        
        ## Kontext:
        {kontext_block}
    """

    return prompt


def answer_question_with_rag(question: str, provider: str, k: int = 10) -> str:
    """
    Führt einen kompletten RAG-Durchlauf durch.
    """
    # 1. Kontext-Chunks zur Frage suchen:
    results = search_similar_chunks(query=question, provider=provider, k=k)

    # 2. Prompt aus Frage + Kontext bauen:
    prompt = build_rag_prompt(question=question, results=results)

    # 3. Chat-Modell aufrufen (LLM):
    client = get_client(provider)
    config = get_config(provider)

    chat_response = client.chat.completions.create(
        model=config["chat_model"],
        messages=[
            {
                "role": "system",
                "content": "Du bist ein hilfreicher RAG-Assistent, der auf Deutsch antwortet.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = chat_response.choices[0].message.content
    return answer

# Helper
def extract_chunks_from_structured_json(structured_json):
    """
    Für chunks aus PDF
    """
    chunks = []

    # 1. Jede Section als eigener Chunk
    for page in structured_json.get("pages", []):
        for sec in page.get("sections", []):
            title = sec.get("title", "")
            content = sec.get("content", "")

            if content:
                # content immer zum Sting machen
                if isinstance(content, list):
                    content = "\n".join([str(c) for c in content])
                else:
                    content = str(content)

                # Titel als Teil des Chunks
                chunk_text = f"{title}\n{content}" if title else content
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)

    # Key-Value-Pairs als Klartext
    kv = structured_json.get("key_value_pairs", {})
    for key, val in kv.items():
        if val:
            chunks.append(f"{key}: {val}")

    return chunks


def ingest_structured_document(structured_json, source: str, provider: str) -> int:
    chunks = extract_chunks_from_structured_json(structured_json)

    if not chunks:
        return 0

    # Embeddings generieren
    embeddings = embed_texts(chunks, provider=provider)

    items = []
    for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        item_id = f"{source}_{idx}"
        metadata = {"source": source, "chunk": idx, "text": chunk_text}
        items.append((item_id, emb, metadata))

    collection = get_collection(provider)
    collection.upsert(items)
    collection.create_index()

    return len(items)
