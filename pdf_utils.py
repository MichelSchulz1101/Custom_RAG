"""
Dieses Modul kombiniert:
- PyPDF2 für normale Texte
- Docling für eingescannten Text und komplexe PDFs
"""

from PyPDF2 import PdfReader
import io
import os
import json
import re
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter


def extract_text_with_pypdf2(pdf_file) -> list[str]:
    """
    Extrahiert Text pro Seite mit PyPDF2 und gibt eine
    Liste zurück. Jeder Eintrag entspricht einer Seite.
    """

    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text is None:
            pages.append("")
        else:
            pages.append(text.strip())

    return pages


def extract_text_with_ocr(pdf_bytes: bytes, lang="de"):
    """
    Uses Docling to extract text from the PDF (including scanned parts).
    Then structures the text using OpenAI.
    """
    # Save bytes to a temporary file because Docling expects a path or URL
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        doc = result.document
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Extract text per page
    page_texts = {}  # page_no -> list of strings

    def traverse(item):
        # If item has children, traverse them
        if hasattr(item, "children") and item.children:
            for child in item.children:
                if hasattr(child, "resolve"):
                    resolved = child.resolve(doc)
                    traverse(resolved)
                else:
                    traverse(child)
            return  # Don't process the group itself as text if we processed children

        # Leaf item (or item without children we care about)
        text_content = ""
        if hasattr(item, "export_to_markdown"):
            try:
                text_content = item.export_to_markdown()
            except:
                if hasattr(item, "text"):
                    text_content = item.text
        elif hasattr(item, "text"):
            text_content = item.text

        if text_content and hasattr(item, "prov") and item.prov:
            # prov is a list of ProvenanceItem
            page_no = item.prov[0].page_no
            if page_no not in page_texts:
                page_texts[page_no] = []

            if isinstance(text_content, list):
                text_content = "\n".join([str(t) for t in text_content])

            page_texts[page_no].append(str(text_content))

    traverse(doc.body)

    structured_pages = []

    # Iterate over pages in order
    sorted_pages = sorted(page_texts.keys())
    for page_number in sorted_pages:
        page_text = "\n".join(page_texts[page_number])

        # Strukturierung der OCR-Ausgabe:
        structured = structure_document(text=page_text)
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


def structure_document(text):
    prompt = f"""
        You are a document structuring system.
        The content is an arbitrary OCR document. 
        Your job is to convert it into a universal JSON format
        that works for ANY document type (letters, contracts, invoices, articles, etc.)
        without assuming a fixed schema.

        The JSON MUST contain ONLY these top-level keys:

        {{
        "document_type": "",
        "title": "",
        "sections": [],
        "tables": [],
        "entities": {{
            "dates": [],
            "names": [],
            "locations": [],
            "organizations": [],
            "amounts": []
        }},
        "key_value_pairs": {{}}
        }}

        Rules:
        - Detect the document type heuristically. Examples include: invoice, receipt, contract, legal letter, academic article, form, notice, report, business letter, manual, etc. 
        Do NOT force any specific type. If uncertain, use "unknown".
        - Split text into logical sections by meaning.
        - Extract entities (NER-like).
        - If text includes something like 'X: Y', treat as key-value pair.
        - If a table-like structure exists, convert it into rows and columns.
        - Preserve original content.

        OCR TEXT:
        {text}
    """

    # Variablen aus .env-Datei laden:
    load_dotenv()

    # Wichtige Konfigurationswerte:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Fehler ausgeben, falls wichtige Werte fehlen:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt (.env-Datei prüfen)!")

    # Modellnamen:
    CHAT_MODEL = "gpt-4o-mini"

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "system",
                "content": "Return ONLY valid JSON without explanation, markdown or comments.",
            },
        ],
    )

    return safe_extract_json(response.choices[0].message.content)


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


# Für Testzwecke:
if __name__ == "__main__":
    # with open("rezept_test.pdf", "rb") as file:
    #     text = extract_text_with_ocr(pdf_bytes=file.read())
    #     print(text)

    text = extract_text_with_pypdf2("rezept_test.pdf")
    print(text)
