import chromadb
from chromadb.utils import embedding_functions
from db.models import db, Chunk
from config import COLLECTION_NAME, OLLAMA_EMBED_MODEL, CHROMA_DB_DIR
import re


def build_and_save_rag_db():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=OLLAMA_EMBED_MODEL,
    )

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,  # type: ignore
        metadata={"hnsw:space": "cosine"},
    )

    query = Chunk.select(Chunk.id, Chunk.raw_text_blob).iterator()

    batch_size = 200
    ids = []
    documents = []

    print("Adding chunks to ChromaDB collection...")
    iter = 0
    for chunk in query:

        ids.append(str(chunk.id))
        raw_text = chunk.get_raw_text()
        text = re.sub(r"[^a-z0-9\s]", "", raw_text.lower())
        text = " ".join(text.split())

        documents.append(text)

        if len(ids) >= batch_size:
            collection.add(ids=ids, documents=documents)
            ids = []
            documents = []
        if iter % 100 == 0:
            print(f"Processed {iter} chunks for ChromaDB...")
        iter += 1

    if ids:
        collection.add(ids=ids, documents=documents)
    print("Finished adding chunks to ChromaDB collection.")


if __name__ == "__main__":
    db.connect()
    build_and_save_rag_db()
    db.close()
