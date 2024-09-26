from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

FOLDER_PATH = "/Users/oscararmas/Desktop/StoriChallenge/"
MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "document_embeddings_v2"
PAGE_TITLE = "Comparison Chatbot: TF-IDF vs Embeddings"


GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}


INTENT_KEYWORDS = {
    "greeting": ["hola", "buenos días", "buenas tardes", "buenas noches"],
    "farewell": ["adiós", "nos vemos", "hasta luego"],
    "thanks": ["gracias", "te lo agradezco"],
    "product_query": ["productos", "tarjetas", "servicios"],
    "irrelevant": ["clima", "películas", "deportes"],
}

INTENT_RESPONSES = {
    "greeting": "¡Hola! ¿En qué puedo asistirte hoy? Estoy aquí para responder preguntas sobre HistoriaCard.",
    "farewell": "¡Adiós! Que tengas un gran día. No dudes en volver si tienes más preguntas.",
    "thanks": "¡De nada! 😊 Estoy aquí para ayudarte cuando lo necesites.",
    "product_query": "HistoriaCard ofrece tarjetas de crédito y débito diseñadas para mejorar el historial crediticio de los usuarios.",
    "irrelevant": "Lo siento, este chat está diseñado para responder preguntas relacionadas con HistoriaCard. ¿Tienes alguna pregunta sobre nuestros productos o servicios?",
}


SAMPLE_QUESTIONS = [
        "¿Cuál es la misión principal de HistoriaCard?", 
        "¿Qué productos financieros ofrece HistoriaCard?", 
        "¿Cómo contribuye HistoriaCard a la inclusión financiera?", 
        "¿Cuáles son las características principales de la tarjeta de crédito HistoriaCard?",
        "¿Qué beneficios adicionales ofrece HistoriaCard con su tarjeta de crédito?", 
        "¿Cómo ayuda HistoriaCard a sus usuarios a mejorar su historial crediticio?", 
        "¿Qué tasa de interés ofrece la tarjeta de crédito HistoriaCard?", 
        "¿Qué funciones de seguridad ofrece HistoriaCard en su app?", 
        "¿Cómo utiliza HistoriaCard la inteligencia artificial en sus servicios?", 
        "¿Cuáles son los principales objetivos de la expansión internacional de HistoriaCard?", 
        "¿Qué alianzas estratégicas ha establecido HistoriaCard?", 
        "¿Cómo contribuye HistoriaCard al ahorro de sus usuarios con la tarjeta de débito?", 
        "¿Qué herramientas educativas ofrece HistoriaCard en su app?", 
        "¿Cómo protege HistoriaCard a los usuarios contra fraudes?", 
        "¿Qué opciones de recompensas ofrece HistoriaCard?", 
        "¿Qué impacto tiene HistoriaCard en la sociedad mexicana?", 
        "¿Qué innovaciones tecnológicas está explorando HistoriaCard?", 
        "¿Cómo ofrece HistoriaCard asistencia en viajes para sus usuarios?", 
        "¿Cuáles son las principales diferencias entre la tarjeta de crédito y la tarjeta de débito HistoriaCard?",
        "¿Cómo acceden los usuarios a la atención al cliente de HistoriaCard?",
        "¿Cuáles son las características principales de la criptomoneda que ofrece HistoriaCard?", # Pregunta no contestable
        "¿Cómo ayuda HistoriaCard a los usuarios a invertir en bolsa?", # Pregunta no contestable
        "¿Qué opciones de hipotecas ofrece HistoriaCard?" # Pregunta no contestable
    ]
