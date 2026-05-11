import re
import json
import zlib
import nltk
from itemadapter import ItemAdapter
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from .db_models import db, Book, Chunk
from .items import BookItem, ChunkItem


class SQLitePipeLine:
    def open_spider(self, spider):
        db.connect()
        db.execute_sql("PRAGMA journal_mode = WAL;")
        db.execute_sql("PRAGMA synchronous = NORMAL;")
        db.create_tables([Book, Chunk])

        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = SnowballStemmer("english")
        self.chunk_buffer = []

    def close_spider(self, spider):
        self.save_bulk_chunks()
        db.close()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        if isinstance(item, BookItem):
            Book.get_or_create(
                id=adapter.get("id"),
                defaults={
                    "title": adapter.get("title"),
                    "author": adapter.get("author"),
                    "image_url": adapter.get("image_url"),
                    "book_url": adapter.get("book_url"),
                },
            )
        elif isinstance(item, ChunkItem):
            self.handle_chunk_item(adapter)

        return item

    def handle_chunk_item(self, adapter):
        raw_text = adapter.get("text")
        if not raw_text:
            return

        first_100_words = " ".join(raw_text.split()[:100])
        compressed_raw = zlib.compress(first_100_words.encode("utf-8"))

        text = re.sub(r"[^a-z0-9\s]", "", raw_text.lower())
        words = [w for w in text.split() if w not in self.stop_words]
        stemmed_words = [self.stemmer.stem(w) for w in words]

        all_tokens = list(stemmed_words)
        for n in range(2, 4):
            grams = ["_".join(g) for g in nltk.ngrams(stemmed_words, n)]
            all_tokens.extend(grams)

        counts = Counter(all_tokens)
        compressed_counts = zlib.compress(json.dumps(counts).encode("utf-8"))

        self.chunk_buffer.append(
            {
                "book_id": adapter.get("book_id"),
                "chunk_url": adapter.get("chunk_url"),
                "compressed_json": compressed_counts,
                "raw_text_blob": compressed_raw,
            }
        )

        if len(self.chunk_buffer) >= 500:
            self.save_bulk_chunks()

    def save_bulk_chunks(self):
        if self.chunk_buffer:
            try:
                with db.atomic():
                    Chunk.insert_many(self.chunk_buffer).execute()
                self.chunk_buffer = []
            except Exception as e:
                print(f"Error bulk saving chunks: {e}")
