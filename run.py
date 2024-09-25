import os

import chromadb
import google.generativeai as genai
import streamlit as st
from chromadb.utils import embedding_functions
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

from app.resources import (
    PDFReader,
    generate_prompt_from_df,
    get_or_create_collection,
    process_and_save_tfidf,
    retrieve_embeddings_documents_df,
    retrieve_tfidf_documents_df,
)

path = os.path.dirname(__file__)
genai.configure(api_key="AIzaSyC4_NC-2eGpg7URKQQqGVQ4mGjZlimhQIQ")
folder_path = "/Users/oscararmas/Desktop/StoriChallenge/"
output_directory = path

st.set_page_config(page_title="Chatbot Comparador: TF-IDF vs Embeddings", layout="wide")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

model_geminis = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

prompt_template = PromptTemplate(
    input_variables=["question", "items"],
    template="""
Te presento una pregunta: {question}

Contexto: 
{items}

Reglas para generar la respuesta:
- Si la pregunta es un saludo o una cortesía (como "hola", "¿cómo estás?", etc.), responde amablemente y redirige la conversación hacia cómo puedes ayudar con preguntas sobre HistoriaCard.
- Proporciona una respuesta clara, concisa y directamente relacionada con la pregunta.
- Si hay múltiples partes en la pregunta, asegúrate de responder a todas ellas.
- Usa un tono educado, profesional, pero amigable.
- Si la información proporcionada no es suficiente para responder de manera precisa, responde: "Lo siento, no tengo suficiente información para responder a esta pregunta de manera precisa basada en los datos proporcionados."
- Si la pregunta no está relacionada con HistoriaCard, responde brevemente y redirige la conversación hacia el propósito de ayudar con preguntas sobre HistoriaCard.
- No agregues información adicional que no esté presente en el contexto.

    """,
)


def detect_intent(message):
    message = message.lower().strip()

    if any(
        greeting in message
        for greeting in ["hola", "buenos días", "buenas tardes", "buenas noches"]
    ):
        return "greeting"
    if any(farewell in message for farewell in ["adiós", "nos vemos", "hasta luego"]):
        return "farewell"
    if any(thank in message for thank in ["gracias", "te lo agradezco"]):
        return "thanks"
    if any(product in message for product in ["productos", "tarjetas", "servicios"]):
        return "product_query"
    if any(irrelevant in message for irrelevant in ["clima", "películas", "deportes"]):
        return "irrelevant"

    return "unknown"


def handle_message(message, items, history, model, collection):
    intent = detect_intent(message)

    if intent in ["greeting", "farewell", "thanks", "product_query", "irrelevant"]:
        responses = {
            "greeting": "¡Hola! ¿En qué puedo asistirte hoy? Estoy aquí para responder preguntas sobre HistoriaCard.",
            "farewell": "¡Adiós! Que tengas un gran día. No dudes en volver si tienes más preguntas.",
            "thanks": "¡De nada! 😊 Estoy aquí para ayudarte cuando lo necesites.",
            "product_query": "HistoriaCard ofrece tarjetas de crédito y débito diseñadas para mejorar el historial crediticio de los usuarios.",
            "irrelevant": "Lo siento, este chat está diseñado para responder preguntas relacionadas con HistoriaCard. ¿Tienes alguna pregunta sobre nuestros productos o servicios?",
        }
        response = responses[intent]
    else:
        result_df_embeddings = retrieve_embeddings_documents_df(
            collection, message, model
        )
        prompt_embb = generate_prompt_from_df(result_df_embeddings, message)
        response = model_geminis.generate_content(prompt_embb).text

    return response


@st.cache_resource
def load_models():
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient()
    collection_name = "document_embeddings_v2"
    collection = client.get_or_create_collection(collection_name)
    return model, collection


@st.cache_data
def load_pdf_and_process():
    pdf_reader = PDFReader(folder_path)
    pdf_texts_dict = pdf_reader.read_pdfs(by_paragraph=True)
    pipeline, vectorizer, document_vectors = process_and_save_tfidf(
        pdf_texts_dict, output_directory
    )
    return pdf_texts_dict, pipeline, vectorizer, document_vectors


model, collection = load_models()
pdf_texts_dict, pipeline, vectorizer, document_vectors = load_pdf_and_process()

st.title("Chatbot Comparador: TF-IDF vs Embeddings con Gemini AI")
st.markdown(
    """
Este chatbot compara las respuestas generadas utilizando TF-IDF y Embeddings con Gemini AI. 
Observa cómo cada método puede producir resultados diferentes para la misma pregunta.
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
        [""]
        + [
            "¿Cuál es la misión principal de HistoriaCard?",
            "¿Cómo utiliza HistoriaCard la inteligencia artificial en sus servicios?",
            "¿Qué productos financieros ofrece HistoriaCard?",
        ],
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

        st.session_state.result_df_tfidf = result_df_tfidf
        st.session_state.result_df_embeddings = retrieve_embeddings_documents_df(
            collection, question, model
        )

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
