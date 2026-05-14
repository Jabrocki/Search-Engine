import nltk
from itemadapter import ItemAdapter
from db.models import db, Book, Chunk
from utils.preprocess_text import TextPreprocessor
from config import NO_CHUNK_TO_SAVE
from .items import BookItem, ChunkItem


class SQLitePipeLine:
    def open_spider(self, spider):
        db.connect()
        db.execute_sql("PRAGMA journal_mode = WAL;")
        db.execute_sql("PRAGMA synchronous = NORMAL;")
        db.create_tables([Book, Chunk])

        nltk.download("stopwords", quiet=True)
        self.preprocessor = TextPreprocessor()
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
        counts = self.preprocessor.preprocess_text(raw_text)
        chunk = Chunk(
            book_id=adapter.get("book_id"),
            chunk_url=adapter.get("chunk_url"),
        )

        chunk.set_json(counts)
        chunk.set_raw_text(raw_text)
        self.chunk_buffer.append(chunk)

        if len(self.chunk_buffer) >= NO_CHUNK_TO_SAVE:
            self.save_bulk_chunks()

    def save_bulk_chunks(self):
        if self.chunk_buffer:
            try:
                with db.atomic():
                    Chunk.bulk_create(self.chunk_buffer, batch_size=NO_CHUNK_TO_SAVE)
                self.chunk_buffer = []
            except Exception as e:
                print(f"Error bulk saving chunks: {e}")
