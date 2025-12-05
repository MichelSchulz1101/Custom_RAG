from dotenv import load_dotenv
import os
import vecs
from openai import OpenAI
from typing import List

# Variablen aus .env-Datei laden:
load_dotenv()

# Wichtige Konfigurationswerte:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Fehler ausgeben, falls wichtige Werte fehlen:
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ist nicht gesetzt (.env-Datei prüfen)!")
if not SUPABASE_DB_URL:
    raise ValueError("SUPABASE_DB_URL ist nicht gesetzt (.env-Datei prüfen)!")

# Modellnamen:
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

# Name der Collection = Eine logische Gruppe von Vektoren (z.B. alle Embeddings der Dokumente):
COLLECTION_NAME = "rag_docs"

# OpenAI-Client, über den wir Embeddings anfragen (auch Chat möglich):
client = OpenAI(api_key=OPENAI_API_KEY)

# Einsteigspunkt für die Verbindung zur Supabase-Datenbank:
vx = vecs.create_client(SUPABASE_DB_URL)

# Collection erstellen oder holen:
# - existiert sie schon, wird sie nur "geöffnet"
# - existiert sie noch nicht, wird sie mit dieser Dimension angelegt:
collection = vx.get_or_create_collection(
    name=COLLECTION_NAME,
    dimension=1536,  # Muss zur Dimensionalität des Embedding-Modells passen
)


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Teilt einen langen Text in kleinere Textstücke (Chunks),
    damit wir pro Chunk ein Embedding erstellen können.

    Parameter:
        text (str): Gesamter Eingabetext (z.B. ein ganzer Artikel oder ein Kapitel)
        max_chars_int (int, optional): Maximale Anzahl an Zeichen pro Chunk

    Rückgabe:
        Liste von von Text-Chunks (Strings)
    """
    chunks = []
    current = []
    current_len = 0

    # Wir trennen grob nach Zeilenumbrüchen:
    for symbol in text.split("\n"):
        # Wenn der akteulle Chunk zu große werden würde, speichern wir ihn ab:
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


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Erzeugt Embeddings für eine Liste von Texten.

    Parameter:
        texts (List[str]): Liste von Strings (Text-Chunks)

    Returns:
        List[List[float]]: Liste von Embeddins (jede Embedding ist eine Liste von floats)
    """
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

    # "response.data" ist eine Liste von Objekten, die jeweils ein Embedding enthalten:
    return [data.embedding for data in response.data]


def ingest_document(raw_text: str, source: str) -> int:
    """
    Nimmt einen Rohtext, zerteilt ihn in Chunks, erstellt Embeddings
    und speichert alles in der Supabase-Collection.

    Parameter:
        raw_text (str): Der komplette Text des Dokuments
        source (str): Eine Kennung für die Quelle z.B. Dateiname ("handbuch_v1)
    """
    # 1. Text in Chunks aufteilen:
    chunks = chunk_text(text=raw_text)
    if not chunks:
        return 0
    print(f"Anzahl Chunks: {len(chunks)}")

    # 2. Embeddings für alle Chunks erzeugen:
    embeddings = embed_texts(texts=chunks)

    # 3. Items für "vecs" vorbereiten:
    # - "vecs" erwartet eine Liste von Tupel: (id, embedding, metadata)
    items = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

        # Eindeutige ID pro Chunk (z.B. "demo_doc_0", "demo_doc_1"...)
        item_id = f"{source}_{i}"

        # Metadaten als JSON-ähnliches Dictionary:
        # - source: Woher kommt der Text?
        # - chunk: Laufende Nummer
        # - text: Der eigentliche Text dieses Chunks
        metadata = {"source": source, "chunk": i, "text": chunk}

        items.append((item_id, emb, metadata))
    print(f"Anzahl der Items für 'upsert': {len(items)}")

    # 4. Alle Items in die Collection schreiben (upsert = insert oder update):
    collection.upsert(items)

    # 5. Index für schnellere Ähnlichkeitssuche erstellen:
    collection.create_index()

    print(f"{len(items)} Chunks von '{source}' gespeichert.")
    return len(items)


def embed_query(query: str) -> List[float]:
    """
    Erzeugt ein einzelnes Embedding für eine Nutzerfrage (Query).
    Nutzt dasselbe Embedding-Modell wie beim Ingest.

    Parameter:
        query (str): Die Nutzerfrage z.B. "Wie groß ist der Durchmesser der Sonne?"

    Returns:
        List[float]: Ein Embedding
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])

    return resp.data[0].embedding


def search_similar_chunks(query: str, k: int = 10):
    """
    Sucht die k ähnlichsten Chunks in der Collection für eine
    gegebene Nutzerfrage (query).

    Parameter:
        query (str): Die Nutzerfrage
        k (int, optional): Anzahl der ähnlichsten Chunks

    Rückgabe:
        Liste von Treffern, wobei jeder Treffer ein Tupel ist:
        (id, score, metadata)
    """
    query_vec = embed_query(query=query)

    result = collection.query(
        data=query_vec,
        limit=k,
        measure="cosine_distance",
        include_metadata=True,
        include_value=True,
    )

    # Eine Liste von Tupel "(id, score, metadata)" ausgeben:
    return result


def build_rag_prompt(question: str, results: List[tuple]) -> str:
    """
    Erzeugt einen vollständigen Prompt für ein RAG-Chatmodell aus der Nutzerfrage und
    den zuvor gefunden Kontext-Chunks.

    Parameter:
        question (str): Die ursprüngliche Nutzerfrage
        results (List[tuple]): Trefferliste der semantischen Suche. JEder Eintrag ist ein Tupel:
        (id, score, metadata)

    Returns:
        str: Ein fertig zusammengesetzter Prompt, der in einer Chat-Anfrage verwendet werden kann.
    """
    kontexte = []
    for vec_id, score, metadata in results:
        kontexte.append(metadata["text"])

    kontext_block = "\n\n---\n\n".join(kontexte)

    prompt = f"""
        ## Allgemein:
        Du bist ein hilfreicher Assistent. Beantworte die Frage ausschließlich mit Hilfe
        des bereitgestellten Kontextes. Wenn die Frage im Kontext nicht klar steht, sage erlich, dass du es nicht weißt.
        
        ## Frage:
        {question}
        
        ## Kontext:
        {kontext_block}
    """

    return prompt


def answer_question_with_rag(question: str, k: int = 10) -> str:
    """
    Führt einen kompletten RAG-Druchlauf durch:
    - Ähnliche Chunks suchen
    - Prompt bauen
    - Chat-Modell aufrufen
    - Antwort als String ausgeben

    Parameter:
        question (str): Die Ausgangsfrage der Nutzers
        k (int, optional): k ähnliche Chunks

    Returns:
        str: Endgültige Antwort auf die Ausgangsfrage der Nutzers
    """
    # 1. Kontext-Chunks zur Frage suchen:
    results = search_similar_chunks(query=question, k=k)

    # 2. Prompt aus Frage + Kontext bauen:
    prompt = build_rag_prompt(question=question, results=results)

    # 3. Chat-Modell aufgrufen (LLM):
    chat_response = client.chat.completions.create(
        model=CHAT_MODEL,
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


def extract_chunks_from_structured_json(structured_json):
    chunks = []

    # 1. Jede Section als eigener Chunk
    for page in structured_json.get("pages", []):
        for sec in page.get("sections", []):
            title = sec.get("title", "")
            content = sec.get("content", "")

            if content:
                # Optional: Titel als Teil des Chunks
                chunk_text = f"{title}\n{content}" if title else content
                chunks.append(chunk_text)

    # 2. OPTIONAL: Key-Value-Pairs als Klartext
    kv = structured_json.get("key_value_pairs", {})
    for key, val in kv.items():
        chunks.append(f"{key}: {val}")

    return chunks


def ingest_structured_document(structured_json, source: str) -> int:
    chunks = extract_chunks_from_structured_json(structured_json)

    if not chunks:
        return 0

    # Embeddings generieren
    embeddings = embed_texts(chunks)

    items = []
    for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        item_id = f"{source}_{idx}"
        metadata = {"source": source, "chunk": idx, "text": chunk_text}
        items.append((item_id, emb, metadata))

    collection.upsert(items)
    collection.create_index()

    return len(items)
