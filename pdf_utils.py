"""
Dieses Moduk kombiniert:
- PyPDF2 für normale Texte
- Tesseract OCR für eingescannten Text
"""

import fitz
from PyPDF2 import PdfReader
import io
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re


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
    Wenn PyPDF2 keinen Text findet, wird die jeweilige Seite als PNG gerendert
    und per Paddle OCR ausgelesen.
    """

    ocr = PaddleOCR(lang=lang)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    structured_pages = []

    for page_number in range(len(doc)):
        page = doc[page_number]

        # Seite rendern (Pixel-Matrix):
        pix = page.get_pixmap(dpi=300)

        # Pixel-Matrix in PNG-Bystes:
        img_bytes = pix.tobytes("png")

        # PNG-Bytes in PIL Image:
        img = Image.open(io.BytesIO(img_bytes))

        # OCR durchführen:
        result = ocr.predict(np.array(img))
        page_text_lines = []
        for res in result:
            if "rec_texts" in res:
                page_text_lines.extend(res["rec_texts"])
        page_text = "\n".join(page_text_lines)

        # Strukturierung der OCR-Ausgabe:
        structured = structure_document(text=page_text)
        structured["page_number"] = page_number + 1
        structured_pages.append(structured)

    # Ganzes Dokument zusammenführen:
    all_page_texts = []
    for page in structured_pages:
        if page.get("sections"):
            for section in page["sections"]:
                content = section.get("content", "")
                if content:
                    all_page_texts.append(content)

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
    CHAT_MODEL = "gpt-4.1-mini"

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
