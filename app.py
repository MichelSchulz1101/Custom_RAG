import streamlit as st
import rag_backend
from PyPDF2 import PdfReader

st.title("RAG-System")

# --- Sidebar: Dokument ingestieren --- #
with st.sidebar:
    st.header("Dokumente ingestieren")
    
    source_name = st.text_input(
        label="Quellen-Name:",
        value="demo_doc"
    )
    
    # Wie sollen die Daten hinzugefügt werden?
    ingest_mode = st.radio(
        label="Wie möchtest du deine Daten hinzufügen?",
        options=("Text einfügen", "PDF hochladen")
    )
    
    if ingest_mode == "Text einfügen":
        raw_text = st.text_area(
            label="Dokument-Text hier einfügen:",
            height=300,
            placeholder="""
                Füge hier deinen Text ein, der in die 
                Wissensdatenbank soll...
            """
        )
    
        if st.button("Ingest starten"):
            if not raw_text.strip():
                st.warning("Bitte zuerst Dokument-Text eingeben!")
            else:
                num_chunks = rag_backend.ingest_document(raw_text, source_name)
                st.success(f"{num_chunks} Chunks unter Quelle '{source_name}' gespeichert.")
    
    else:
        uploaded_pdf = st.file_uploader(
            label="PDF-Datei auswählen:",
            type=["pdf"]
        )
        
        if st.button("Ingest starten"):
            try:
                # PDF einlesen und Text extrahieren:
                reader = PdfReader(uploaded_pdf)
                full_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                
                if not full_text.strip():
                    st.error("Kein Text in der PDF gefunden!") 
                else:
                    num_chunks = rag_backend.ingest_document(
                        raw_text=full_text,
                        source=source_name
                    )
                    st.success(
                        f"{num_chunks} Chunks aus PDF unter Quelle '{source_name}'"
                        "gespeichert."
                    )
                          
            except Exception as e:
                st.error(f"Fehler beim Lesen der PDF: {e}")
        
                
st.markdown("## Chat mit dem RAG")

if st.button(label="Chat löschen"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chat zurückgesetzt!"
        }
    ]
    
    st.success("Chat-Verlauf wurde gelöscht!")
    
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Du kannst mir jetzt Fragen zu deinen Dokumenten stellen."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(name=message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Was möchtest du wissen?")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

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
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
