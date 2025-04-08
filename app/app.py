import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

# --- Config ---
CHROMA_PATH = ".\\chroma_db"

# --- Initialisation ---
@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="rag_collection")
    return collection

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_models():
    return {
        "RoBERTa QA (SQuAD2)": pipeline("question-answering", model="deepset/roberta-base-squad2"),
        "DistilBERT QA": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
        "BERT Base QA": pipeline("question-answering", model="bert-base-uncased")
    }

def retrieve_context(question: str, collection, k: int = 3) -> str:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = embedding_model.encode([question])[0].tolist()
    
    results = collection.query(query_embeddings=[question_embedding], n_results=k)
    metadatas = results.get("metadatas", [[]])[0]  # Liste de dicts
    contexts = [meta["text"] for meta in metadatas if "text" in meta]
    
    return "\n".join(contexts) if contexts else "Aucun contexte trouvé."

# --- App UI ---
st.title("📚 RAG Demo - BSc Student Handbook")
st.markdown("Pose une question sur le programme — compare les réponses de différents modèles.")

question = st.text_input("Your question :", placeholder="What is the name of the program in data science?")

if question:
    collection = load_chroma()

    context = retrieve_context(question, collection)

    st.markdown("### Contexte found")
    st.write(context)

    # Appliquer les 3 modèles
    st.markdown("### Answer of the model :")
    models = load_models()

    cols = st.columns(len(models))
    for (name, pipe), col in zip(models.items(), cols):
        try:
            response = pipe(question=question, context=context)
            col.markdown(f"**{name}**")
            col.success(response["answer"])
        except Exception as e:
            col.markdown(f"**{name}**")
            col.error(f"Erreur: {e}")

