from search_engine.bm25_model import BM25SearchModel
from utils.cosine_similarity import cosine_similarity
import joblib


class BM25SvdSearchModel(BM25SearchModel):
    def __init__(self, svd_file: str):
        super().__init__()
        svd_bundle = joblib.load(svd_file)
        self.index_matrix = svd_bundle["index_matrix"]
        self.chunk_ids: list = svd_bundle["chunk_ids"]
        self.svd_model = svd_bundle["svd_model"]

    def search_query(self, query, top_n) -> list[tuple[int, float]]:
        processed_query = self.text_preprocessor.preprocess_text(query)
        filtered_query = {
            key: count for key, count in processed_query.items() if key in self.vocab
        }

        if not filtered_query:
            return []

        query_vec = self.vectorizer.transform([filtered_query])
        query_bm25 = self.bm25.transform(query_vec)
        query_svd = self.svd_model.transform(query_bm25)

        similarities = cosine_similarity(query_svd, self.index_matrix)
        top_indices = similarities.argsort()[::-1][:top_n]

        return [
            (int(self.chunk_ids[idx]), float(similarities[idx])) for idx in top_indices
        ]
