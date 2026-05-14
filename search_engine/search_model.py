from abc import ABC, abstractmethod
from db.models import Chunk, Book, db


class SearchModel(ABC):
    def __init__(self):
        db.connect()

    def __del__(self):
        db.close()

    @abstractmethod
    def search_query(self, query, top_n) -> list[tuple[int, float]]:
        """
        Returns list of chunk_ids and their similarity scores
        """
        pass

    def search(self, query, top_n=5) -> list[dict]:
        search_results = self.search_query(query, top_n)
        results = []
        for chunk_id, score in search_results:
            chunk = Chunk.get(Chunk.id == chunk_id)
            book = Book.get(Book.id == chunk.book_id)
            results.append(
                {
                    "book_title": book.title,
                    "book_author": book.author,
                    "book_url": book.book_url,
                    "book_image_url": book.image_url,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk.get_raw_text_100_first_words(),
                    "chunk_url": chunk.chunk_url,
                    "score": score,
                }
            )
        return results
