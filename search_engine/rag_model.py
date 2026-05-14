import chromadb
from chromadb.utils import embedding_functions
from search_engine.search_model import SearchModel
from config import CHROMA_DB_DIR, COLLECTION_NAME, OLLAMA_EMBED_MODEL


class RAGSearchModel(SearchModel):
    def __init__(self, chroma_path: str = str(CHROMA_DB_DIR)):
        super().__init__()

        self.client = chromadb.PersistentClient(path=chroma_path)

        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL,
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME, embedding_function=self.ollama_ef  # type: ignore
        )

    def search_query(self, query: str, top_n: int = 5) -> list[tuple[int, float]]:
        results = self.collection.query(query_texts=[query], n_results=top_n)

        if (
            not results["ids"]
            or not results["ids"][0]
            or not results["distances"]
            or not results["distances"][0]
        ):
            return []

        chunk_ids_str = results["ids"][0]
        distances = results["distances"][0]

        formatted_results = []
        for string_id, distance in zip(chunk_ids_str, distances):
            chunk_id = int(string_id)
            similarity_score = 1.0 - distance
            formatted_results.append((chunk_id, float(similarity_score)))

        return formatted_results
