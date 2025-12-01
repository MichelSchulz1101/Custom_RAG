import streamlit as st
import rag_backend

st.title("RAG-System")

with st.sidebar:
    source_name = st.text_input(
        label="Quellen-Name:",
        value="demo_doc"         
    )

    raw_text = st.text_area(
        label="Dokument-Text hier einfügen:",
        height=300,
        placeholder="""Füge hier deinen Text ein, der in die Wissensdatenbank soll...
        """
    )

    if st.button("Ingest starten"):
        if not raw_text.strip():
            st.warning("Bitte zuerst Dokument-Text eingeben!")
        else: 
            num_chunks = rag_backend.ingest_document(raw_text, source_name)
            st.success(f"{num_chunks} Chunks unter Quelle '{source_name}' gespeichert.")