from search_engine.tfidf_model import TfidfSearchModel
import joblib
from utils.cosine_similarity import cosine_similarity
from config import RAW_MATRIX_TFIDF


class TfidfRawSearchModel(TfidfSearchModel):
    def __init__(self):
        super().__init__()
        tfidf_bundle = joblib.load(RAW_MATRIX_TFIDF)
        self.index_matrix = tfidf_bundle["index_matrix"]
        self.chunk_ids: list = tfidf_bundle["chunk_ids"]

    def search_query(self, query, top_n) -> list[tuple[int, float]]:
        processed_query = self.text_preprocessor.preprocess_text(query)
        # Filter tokens to those in vocab
        filtered_query = {
            key: count for key, count in processed_query.items() if key in self.vocab
        }

        if not filtered_query:
            return []

        query_vec = self.vectorizer.transform([filtered_query])
        query_tfidf = self.tfidf.transform(query_vec)

        similarities = cosine_similarity(query_tfidf, self.index_matrix)
        top_indices = similarities.argsort()[::-1][:top_n]

        return [
            (int(self.chunk_ids[idx]), float(similarities[idx])) for idx in top_indices
        ]
