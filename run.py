from app.resources import PDFReader, process_and_save_tfidf, retrieve_tfidf_documents_df, get_or_create_collection, retrieve_embeddings_documents_df, generate_prompt_from_df
import google.generativeai as genai
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import streamlit as st
path = os.path.dirname(__file__)

# Uso de la clase
genai.configure(api_key="AIzaSyC4_NC-2eGpg7URKQQqGVQ4mGjZlimhQIQ")
folder_path = "/Users/oscararmas/Desktop/StoriChallenge/"
pdf_reader = PDFReader(folder_path)
pdf_texts_dict = pdf_reader.read_pdfs(by_paragraph=True)
QUERY_QUESTION = "¿Cuál es la tasa de retención de clientes de HistoriaCard después del primer año de uso, y cómo se compara con el promedio de la industria fintech en México?"
output_directory = path
st.set_page_config(page_title="Chatbot Comparador: TF-IDF vs Embeddings", layout="wide")

@st.cache_resource
def load_models():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient()
    collection_name = "document_embeddings_v2"
    collection = client.get_or_create_collection(collection_name)
    return model, collection

@st.cache_data
def load_pdf_and_process():
    folder_path = "/Users/oscararmas/Desktop/StoriChallenge/"
    pdf_reader = PDFReader(folder_path)
    pdf_texts_dict = pdf_reader.read_pdfs(by_paragraph=True)
    pipeline, vectorizer, document_vectors = process_and_save_tfidf(pdf_texts_dict, output_directory)
    return pdf_texts_dict, pipeline, vectorizer, document_vectors

model_geminis = genai.GenerativeModel("gemini-1.5-flash")
model, collection = load_models()
pdf_texts_dict, pipeline, vectorizer, document_vectors = load_pdf_and_process()

# Título y descripción
st.title("Chatbot Comparador: TF-IDF vs Embeddings")
st.markdown("""
Este chatbot compara las respuestas generadas utilizando dos métodos diferentes de recuperación de información: 
TF-IDF y Embeddings. Observa cómo cada método puede producir resultados diferentes para la misma pregunta.
""")

# Inicializar el historial del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Crear columnas para el chat y las tablas de similitud
col1, col2 = st.columns([3, 2])

with col1:
    # Mostrar el historial del chat con scroll
    st.write("### Historial del Chat")
    chat_container = st.container()
    with chat_container:
        # Añadir scroll al historial del chat
        st.markdown("""
        <div style="max-height: 400px; overflow-y: auto; padding-right: 10px;">
        """, unsafe_allow_html=True)
        
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                st.markdown(f"**Usuario:** {chat['content']}")
            elif chat["role"] == "tfidf":
                st.markdown(f"<div style='background-color: #e0f7fa; padding: 10px; border-radius: 5px;'><strong>TF-IDF:</strong> {chat['content']}</div>", unsafe_allow_html=True)
            elif chat["role"] == "embeddings":
                st.markdown(f"<div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px;'><strong>Embeddings:</strong> {chat['content']}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Input del usuario
    user_input = st.text_input("Escribe tu pregunta aquí:", key="user_input")
    selected_question = st.selectbox("O selecciona una pregunta:", [""] + [
        "¿Cuál es la misión principal de HistoriaCard?", 
        "¿Cómo utiliza HistoriaCard la inteligencia artificial en sus servicios?", 
        "¿Qué productos financieros ofrece HistoriaCard?"
    ])
    
    if st.button("Enviar"):
        question = user_input if user_input else selected_question
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Procesar con TF-IDF
        result_df_tfidf = retrieve_tfidf_documents_df(pdf_texts_dict, question, document_vectors, vectorizer, pipeline, top_k=3)
        prompt_tfidf = generate_prompt_from_df(result_df_tfidf, question)
        response_tf_idf = model_geminis.generate_content(prompt_tfidf).text
        st.session_state.chat_history.append({"role": "tfidf", "content": response_tf_idf})

        # Procesar con Embeddings
        result_df_embeddings = retrieve_embeddings_documents_df(collection, question, model)
        prompt_embb = generate_prompt_from_df(result_df_embeddings, question)
        response_emb = model_geminis.generate_content(prompt_embb).text
        st.session_state.chat_history.append({"role": "embeddings", "content": response_emb})

        # Guardar los resultados en session_state
        st.session_state.result_df_tfidf = result_df_tfidf
        st.session_state.result_df_embeddings = result_df_embeddings

        st.rerun()

with col2:
    if "result_df_tfidf" in st.session_state and "result_df_embeddings" in st.session_state:
        st.subheader("Documentos Relevantes")
        tab1, tab2 = st.tabs(["TF-IDF", "Embeddings"])
        
        with tab1:
            st.dataframe(st.session_state.result_df_tfidf, use_container_width=True)
        
        with tab2:
            st.dataframe(st.session_state.result_df_embeddings, use_container_width=True)

# Añadir un poco de CSS para mejorar el chat con scroll
st.markdown("""
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
""", unsafe_allow_html=True)