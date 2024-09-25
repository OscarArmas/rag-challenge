import os
import PyPDF2
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import PromptTemplate
from typing import Dict, Tuple, Any

import pandas as pd
nltk.download('stopwords')

def process_and_save_tfidf(pdf_texts_dict: Dict[str, str], output_dir: str) -> Tuple[Any, TfidfVectorizer, Any]:
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    vectors_path = os.path.join(output_dir, 'document_vectors.pkl')
    
    if os.path.exists(pipeline_path) and os.path.exists(vectorizer_path) and os.path.exists(vectors_path):
        print("Artefactos encontrados. Cargando...")
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(vectors_path, 'rb') as f:
            document_vectors = pickle.load(f)
        print("Artefactos cargados exitosamente.")
    else:
        print("Artefactos no encontrados. Creando nuevos...")
        pipeline = TextPreprocessingPipeline(language='spanish')
        documents_cleaned = [pipeline.transform(doc) for doc in pdf_texts_dict.values()]
        
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(documents_cleaned)
        
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        with open(vectors_path, 'wb') as f:
            pickle.dump(document_vectors, f)
        
        print("Nuevos artefactos creados y guardados exitosamente.")
    
    return pipeline, vectorizer, document_vectors

def retrieve_embeddings_documents_df(collection, query, model, n_results=3):
    query_embedding = model.encode([query])
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    documents = results['documents'][0]
    distances = results['distances'][0]
    
    df = pd.DataFrame({
        'Document': documents,
        'Similarity Score': distances
    })
    
    df = df.sort_values('Similarity Score', ascending=True).reset_index(drop=True)
    
    return df
    
def retrieve_tfidf_documents_df(data, query, document_vectors, vectorizer, pipeline, top_k=2):
    query_cleaned = pipeline.transform(query)
    query_vector = vectorizer.transform([query_cleaned])
    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
    
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    result = [(data[i], similarity_scores[i]) for i in top_indices]
    
    df = pd.DataFrame(result, columns=["Document", "Similarity Score"])
    return df

def get_or_create_collection(client, collection_name):
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Colección '{collection_name}' cargada exitosamente.")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Colección '{collection_name}' creada exitosamente.")
    return collection

class TextPreprocessingPipeline:
    def __init__(self, language='spanish'):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        self.stopwords = set(stopwords.words(language))
    
    def clean_text(self, text):
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales
        text = re.sub(r'[^a-záéíóúñü\s]', '', text)
        # Eliminar espacios adicionales
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        # Eliminar las stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords]
        return ' '.join(words)
    
    def stem_text(self, text):
        # Aplicar stemming
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def transform(self, text):
        # Aplicar todas las transformaciones
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stem_text(text)
        return text

    
def generate_prompt_from_df(df: pd.DataFrame, question: str) -> str:
    items = df['Document'].tolist()
    
    prompt_template = PromptTemplate(
        input_variables=["question", "items"],
        template="""
        Contesta la siguiente pregunta: {question}
        

        Basado en la siguiente lista de información:

        {items}
        
        con las condiciones:
        - Respuesta breves
        - Solo haz referencia a el texto, no agregues nada mas.
        si no hay informacion relevante, contesta:
        'Informacion no encontrada'
        """
    )
    
    prompt = prompt_template.format(
        question=question,
        items="\n\n".join(items)
    )
    
    return prompt

class PDFReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def read_pdfs(self, by_paragraph=False):
        all_texts = []
        
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text:
                            if by_paragraph:
                                paragraphs = text.split(".")
                                for paragraph in paragraphs:
                                    cleaned_paragraph = paragraph.strip()
                                    if cleaned_paragraph:
                                        all_texts.append(cleaned_paragraph + ".")
                            else:
                                all_texts.append(text.strip())
                        else:
                            all_texts.append("")

        return all_texts
