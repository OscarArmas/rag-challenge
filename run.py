import os
import chromadb
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
from app.config import *

from app.resources import (
    PDFReader,
    generate_prompt_from_df,
    process_and_save_tfidf,
    retrieve_embeddings_documents_df,
    retrieve_tfidf_documents_df,
)

path = os.path.dirname(__file__)
output_directory = path

genai.configure(api_key=GOOGLE_API_KEY)
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

model_geminis = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=GENERATION_CONFIG,
    system_instruction="Eres un asistente AI que responde preguntas sobre HistoriaCard. Todas tus respuestas deben ser como miembro perteneciente a HistoriaCard"
)

def detect_intent(message):
    message = message.lower().strip()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            return intent
    return "unknown"

def handle_message(message, items, history, model, collection):
    intent = detect_intent(message)
    if intent in INTENT_RESPONSES:
        return INTENT_RESPONSES[intent]
    else:
        result_df_embeddings = retrieve_embeddings_documents_df(collection, message, model)
        prompt_embb = generate_prompt_from_df(result_df_embeddings, message)
        return model_geminis.generate_content(prompt_embb).text

@st.cache_resource
def load_models():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return model, collection

@st.cache_data
def load_pdf_and_process():
    pdf_reader = PDFReader(FOLDER_PATH)
    pdf_texts_dict = pdf_reader.read_pdfs(by_paragraph=True)
    pipeline, vectorizer, document_vectors = process_and_save_tfidf(pdf_texts_dict, output_directory)
    return pdf_texts_dict, pipeline, vectorizer, document_vectors

model, collection = load_models()
pdf_texts_dict, pipeline, vectorizer, document_vectors = load_pdf_and_process()

st.title(PAGE_TITLE)
st.markdown(
    """
TF-IDF vs Embeddings with Gemini AI.
Observe how each method can produce different results for the same question.
"""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([3, 2])

with col1:
    st.write("### Historial del Chat")
    chat_container = st.container()
    with chat_container:
        st.markdown(
            """
        <div style="max-height: 400px; overflow-y: auto; padding-right: 10px;">
        """,
            unsafe_allow_html=True,
        )

        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**Usuario:** {chat['content']}")
            elif chat["role"] == "tfidf":
                st.markdown(
                    f"<div style='background-color: #e0f7fa; padding: 10px; border-radius: 5px;'><strong>TF-IDF:</strong> {chat['content']}</div>",
                    unsafe_allow_html=True,
                )
            elif chat["role"] == "embeddings":
                st.markdown(
                    f"<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px;'><strong>Embeddings:</strong> {chat['content']}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.text_input("Escribe tu pregunta aquí:", key="user_input")
    selected_question = st.selectbox(
        "O selecciona una pregunta:",
        [""] + SAMPLE_QUESTIONS,
    )

    if st.button("Enviar"):
        question = user_input if user_input else selected_question
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Procesar con TF-IDF
        result_df_tfidf = retrieve_tfidf_documents_df(
            pdf_texts_dict, question, document_vectors, vectorizer, pipeline, top_k=3
        )
        prompt_tfidf = generate_prompt_from_df(result_df_tfidf, question)
        response_tf_idf = model_geminis.generate_content(prompt_tfidf).text
        st.session_state.chat_history.append(
            {"role": "tfidf", "content": response_tf_idf}
        )

        # Procesar con Embeddings y Gemini AI
        response_emb = handle_message(
            question, "", st.session_state.chat_history, model, collection
        )
        st.session_state.chat_history.append(
            {"role": "embeddings", "content": response_emb}
        )
        retrieve_df_embeddings= retrieve_embeddings_documents_df(
                    collection, question, model
        )
        st.session_state.result_df_tfidf = result_df_tfidf
        st.session_state.result_df_embeddings = retrieve_df_embeddings
        print(retrieve_df_embeddings)

        st.rerun()

with col2:
    if (
        "result_df_tfidf" in st.session_state
        and "result_df_embeddings" in st.session_state
    ):
        st.subheader("Documentos Relevantes")
        tab1, tab2 = st.tabs(["TF-IDF", "Embeddings"])

        with tab1:
            st.dataframe(st.session_state.result_df_tfidf, use_container_width=True)

        with tab2:
            st.dataframe(
                st.session_state.result_df_embeddings, use_container_width=True
            )

st.markdown(
    """
<style>
    .stTextInput > div > div > input {
        caret-color: #4CAF50;
    }
    .stButton > button {
        width: 100%;
    }
    div[style*="max-height"] {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)
