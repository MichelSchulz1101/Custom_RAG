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

st.markdown("## Chat mit dem RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistent",
            "content": "Hi! Du kannst mir jetzt Fragen zu deinen Dokumenten stellen "
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(name="assistant"):
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
