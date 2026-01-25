# app/db.py
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ✅ Use stable, cross-platform embeddings
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

vectordb = PersistentClient(path="chroma_store").get_or_create_collection(
    name="exam_chunks",
    embedding_function=embedding_function
)
