"""
Dieses Modul kümmert sich um das Einlesen und Strukturieren von PDFs.
Es nutzt 'Docling' für OCR und Layout-Erkennung.
"""

import os
import json
import re
import tempfile
from docling.document_converter import DocumentConverter
import rag_backend


def extract_text_with_ocr(pdf_bytes: bytes, provider: str = "Local (Ollama)", lang="de"):
    """
    Nutzt Docling, um Text aus dem PDF zu extrahieren (auch gescannt).
    Danach wird der Text mit einem LLM strukturiert.
    """
    # Temporäre Datei erstellen, da Docling einen Dateipfad benötigt
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        doc = result.document
    finally:
        # Aufräumen: Temporäre Datei löschen
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Text pro Seite sammeln
    page_texts = {}  # Seite -> Liste von Texten

    def traverse(item):
        """Rekursive Funktion, um durch die Dokumenten-Struktur zu gehen."""
        # Wenn das Element Kinder hat, gehe tiefer
        if hasattr(item, "children") and item.children:
            for child in item.children:
                if hasattr(child, "resolve"):
                    resolved = child.resolve(doc)
                    traverse(resolved)
                else:
                    traverse(child)
            return

        # Text-Inhalt extrahieren
        text_content = ""
        if hasattr(item, "export_to_markdown"):
            try:
                text_content = item.export_to_markdown()
            except:
                if hasattr(item, "text"):
                    text_content = item.text
        elif hasattr(item, "text"):
            text_content = item.text

        # Wenn Text gefunden wurde, der Seite zuordnen
        if text_content and hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no
            if page_no not in page_texts:
                page_texts[page_no] = []

            if isinstance(text_content, list):
                text_content = "\n".join([str(t) for t in text_content])

            page_texts[page_no].append(str(text_content))

    # Start der Rekursion
    traverse(doc.body)

    structured_pages = []

    # Iterate over pages in order
    sorted_pages = sorted(page_texts.keys())
    for page_number in sorted_pages:
        page_text = "\n".join(page_texts[page_number])

        # Strukturierung der OCR-Ausgabe:
        structured = structure_document(text=page_text, provider=provider)
        structured["page_number"] = page_number  # page_number is 1-based from docling
        structured_pages.append(structured)

    # Ganzes Dokument zusammenführen:
    all_page_texts = []
    for page in structured_pages:
        if page.get("sections"):
            for section in page["sections"]:
                content = section.get("content", "")
                if content:
                    if isinstance(content, list):
                        content = "\n".join([str(c) for c in content])
                    all_page_texts.append(str(content))

    full_text = "\n".join(all_page_texts)

    return {"pages": structured_pages, "merged_text": full_text}


import rag_backend


def structure_document(text, provider="Local (Ollama)"):
    if not text or not text.strip():
        return {
            "document_type": "unknown",
            "title": "",
            "sections": [],
            "tables": [],
            "entities": {
                "dates": [],
                "names": [],
                "locations": [],
                "organizations": [],
                "amounts": [],
            },
            "key_value_pairs": {},
        }

    system_prompt = """
        You are a document structuring system.
        Your job is to convert OCR text into a universal JSON format.
        The JSON MUST contain ONLY these top-level keys:
        {
        "document_type": "string",
        "title": "string",
        "sections": [{"title": "string", "content": "string"}],
        "tables": [],
        "entities": {
            "dates": [],
            "names": [],
            "locations": [],
            "organizations": [],
            "amounts": []
        },
        "key_value_pairs": {}
        }
        Return ONLY valid JSON.
    """

    user_prompt = f"""
        Analyze the following text and structure it according to the rules.
        
        OCR TEXT:
        {text}
    """

    # Client und Config holen:
    client = rag_backend.get_client(provider)
    config = rag_backend.get_config(provider)

    try:
        response = client.chat.completions.create(
            model=config["chat_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content
        return safe_extract_json(content)

    except Exception as e:
        print(f"Error in structure_document: {e}")
        # Fallback: Return unstructured text as one section
        return {
            "document_type": "unknown",
            "title": "Error processing",
            "sections": [{"title": "Raw Text", "content": text}],
            "tables": [],
            "entities": {},
            "key_value_pairs": {},
        }


def safe_extract_json(text):
    """
    Extrahiert JSON auch aus Text, der ChatGPT drum herum schreibt,
    oder wenn das JSON nicht ganz oben steht.
    """
    if not text or not isinstance(text, str):
        raise ValueError("LLM output is empty or not a string")

    # JSON irgendwo im Text finden
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"Kein JSON gefunden in der Antwort:\n{text}")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"JSON war nicht parsebar:\n{json_str}") from e
