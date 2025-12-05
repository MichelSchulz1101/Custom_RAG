import streamlit as st
import rag_backend
from PyPDF2 import PdfReader
import pdf_utils

st.title("RAG-System")

# --- Sidebar: Dokument ingestieren --- #
with st.sidebar:
    st.header("Dokumente ingestieren")

    uploaded_pdf = None

    if "source_name" not in st.session_state:
        st.session_state["source_name"] = "demo_doc"
    if "auto_source_name" not in st.session_state:
        st.session_state["auto_source_name"] = None

    # Wie sollen die Daten hinzugefügt werden?
    ingest_mode = st.radio(
        label="Wie möchtest du deine Daten hinzufügen?",
        options=("Text einfügen", "PDF hochladen"),
    )

    if ingest_mode == "Text einfügen":
        raw_text = st.text_area(
            label="Dokument-Text hier einfügen:",
            height=300,
            placeholder="""
                Füge hier deinen Text ein, der in die 
                Wissensdatenbank soll...
            """,
        )

        source_name = st.text_input(label="Quellen-Name:", key="source_name")

        if st.button("Ingest starten"):
            if not raw_text.strip():
                st.warning("Bitte zuerst Dokument-Text eingeben!")
            else:
                num_chunks = rag_backend.ingest_document(raw_text, source_name)
                st.success(
                    f"{num_chunks} Chunks unter Quelle '{source_name}' gespeichert."
                )

    else:
        uploaded_pdf = st.file_uploader(label="PDF-Datei auswählen:", type=["pdf"])

        if uploaded_pdf is not None:
            # Dateiname als Vorschlag verwenden:
            filename = uploaded_pdf.name
            dot_index = filename.rfind(".")
            if dot_index != -1:
                auto_source_name = filename[:dot_index]
            else:
                auto_source_name = filename

            if st.session_state["source_name"] == "demo_doc":
                st.session_state["source_name"] = auto_source_name

            st.info(
                f"Quellen-Name wird automatisch auf **'{st.session_state['source_name']}'** gesetzt."
            )

        source_name = st.text_input(label="Quellen-Name:", key="source_name")

        if st.button("Ingest starten"):
            try:
                # PDF einlesen und Text extrahieren:
                pdf_bytes = uploaded_pdf.read()
                structured = pdf_utils.extract_text_with_ocr(pdf_bytes)
                num_chunks = rag_backend.ingest_structured_document(
                    structured_json=structured, source=source_name
                )
                st.success(
                    f"{num_chunks} Chunks aus PDF unter Quelle '{st.session_state['source_name']}'"
                    "gespeichert."
                )

            except Exception as e:
                st.error(f"Fehler beim Lesen der PDF: {e}")


st.markdown("## Chat mit dem RAG")

if st.button(label="Chat löschen"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat zurückgesetzt!"}]

    st.success("Chat-Verlauf wurde gelöscht!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Du kannst mir jetzt Fragen zu deinen Dokumenten stellen.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Was möchtest du wissen?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # User-Nachricht anzeigen:
    with st.chat_message(name="user"):
        st.markdown(user_input)

    # RAG-Antwort anzeigen:
    with st.chat_message(name="assistant"):
        with st.spinner("Denke nach..."):
            try:
                answer = rag_backend.answer_question_with_rag(question=user_input)
            except Exception as e:
                answer = f"Es ist ein Fehler aufgetreten: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
