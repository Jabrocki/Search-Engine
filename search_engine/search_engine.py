from collections import Counter
import joblib
import nltk
import zlib
from utils import load_vocabulary
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from db_models import db, Chunk, Book
import re

SHARED_DATA = joblib.load("shared_models.joblib")
VECTORIZER: DictVectorizer = SHARED_DATA["vectorizer"]
TFIDF: TfidfTransformer = SHARED_DATA["tfidf"]


class SearchEngine:
    def __init__(self, index_matrix_filename: str):
        db.connect()

        svd_bundle = joblib.load(index_matrix_filename)
        self.index_matrix = svd_bundle["index_matrix"]
        self.chunk_ids: list = svd_bundle["chunk_ids"]
        self.svd_model: TruncatedSVD = svd_bundle["svd_model"]
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))
        self.vocab = load_vocabulary()

    # TODO: create global function for text preprocessing
    def search(self, query: str, top_n=10):
        text = re.sub(r"[^a-z0-9\s]", "", query.lower())

        words = [text for text in text.split() if text not in self.stop_words]
        stemmed_words = [self.stemmer.stem(word) for word in words]
        all_tokens = list(stemmed_words)

        for n in range(2, 4):
            grams = ["_".join(g) for g in nltk.ngrams(stemmed_words, n)]
            all_tokens.extend(grams)
        all_tokens = [token for token in all_tokens if token in self.vocab]
        if len(all_tokens) == 0:
            return []

        counts = Counter(all_tokens)

        query_vec = VECTORIZER.transform([counts])
        query_tfidf = TFIDF.transform(query_vec)
        query_svd = self.svd_model.transform(query_tfidf)

        similarities = cosine_similarity(query_svd, self.index_matrix).flatten()
        related_indices = similarities.argsort()[::-1][:top_n]

        results = []
        for idx in related_indices:
            chunk_id = self.chunk_ids[idx]
            chunk_url = Chunk.get(Chunk.id == chunk_id).chunk_url

            blob_data = Chunk.get(Chunk.id == chunk_id).raw_text_blob
            try:
                decompressed_data = zlib.decompress(blob_data)
                chunk_text = decompressed_data.decode("utf-8")
            except zlib.error:
                chunk_text = blob_data.decode("utf-8")

            book_id = Chunk.get(Chunk.id == chunk_id).book_id
            book_title = Book.get(Book.id == book_id).title
            book_author = Book.get(Book.id == book_id).author
            book_url = Book.get(Book.id == book_id).book_url
            book_image_url = Book.get(Book.id == book_id).image_url

            results.append(
                {
                    "chunk_url": chunk_url,
                    "chunk_text": chunk_text,
                    "book_title": book_title,
                    "book_author": book_author,
                    "book_url": book_url,
                    "book_image_url": book_image_url,
                    "score": similarities[idx],
                }
            )

        return results


### Util function for preaty prints for debug


def print_results(query, results):
    print("=" * 40)
    print(f"Search Query: {query}")
    print(f"Top {len(results)} search results:")
    if len(results) == 0:
        print("No results found.")
        return
    for result in results:
        print(f"Chunk URL: {result['chunk_url']}")
        print(f"Book Title: {result['book_title']}")
        print(f"Book Author: {result['book_author']}")
        print(f"Chunk Text: {result['chunk_text']}")
        print(f"Book URL: {result['book_url']}")
        print(f"Book Image URL: {result['book_image_url']}")
        print(f"Similarity Score: {result['score']:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    search_engine = SearchEngine("matrix_svd_300.joblib")
    while True:
        user_query = input("Enter your search query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        search_results = search_engine.search(user_query, top_n=5)
        print_results(user_query, search_results)
