from search_engine.bm25_model import BM25SearchModel
from utils.cosine_similarity import cosine_similarity
import joblib


class BM25RawSearchModel(BM25SearchModel):
    def __init__(self):
        super().__init__()
        bm25_bundle = joblib.load("bm25_raw_index.joblib")
        self.index_matrix = bm25_bundle["index_matrix"]
        self.chunk_ids: list = bm25_bundle["chunk_ids"]

    def search_query(self, query, top_n) -> list[tuple[int, float]]:
        processed_query = self.text_preprocessor.preprocess_text(query)
        filtered_query = {
            key: count for key, count in processed_query.items() if key in self.vocab
        }

        if not filtered_query:
            return []

        query_vec = self.vectorizer.transform([filtered_query])
        query_bm25 = self.bm25.transform(query_vec)

        similarities = cosine_similarity(query_bm25, self.index_matrix)
        top_indices = similarities.argsort()[::-1][:top_n]

        return [
            (int(self.chunk_ids[idx]), float(similarities[idx])) for idx in top_indices
        ]
