from peewee import SqliteDatabase, Model, IntegerField, TextField, BlobField, AutoField
import zlib
import json
from config import DB_PATH

db = SqliteDatabase(DB_PATH)


class Book(Model):
    id = IntegerField(unique=True)
    title = TextField()
    author = TextField(null=True)
    image_url = TextField(null=True)
    book_url = TextField(null=True)

    class Meta:
        database = db


class Chunk(Model):
    id = AutoField()
    book_id = IntegerField()
    chunk_url = TextField()
    compressed_json = BlobField()
    raw_text_blob = BlobField()

    class Meta:
        database = db

    def set_json(self, data_dict: dict):
        json_data = json.dumps(data_dict).encode("utf-8")
        self.compressed_json = zlib.compress(json_data)

    def set_raw_text(self, text: str):
        text_data = text.encode("utf-8")
        self.raw_text_blob = zlib.compress(text_data)

    def get_json(self) -> dict:
        raw_data = bytes(self.compressed_json)
        decompressed = zlib.decompress(raw_data).decode("utf-8")
        return json.loads(decompressed)

    def get_raw_text(self) -> str:
        raw_data = bytes(self.raw_text_blob)
        decompressed = zlib.decompress(raw_data).decode("utf-8")
        return decompressed

    def get_raw_text_100_first_words(self) -> str:
        raw_text = self.get_raw_text()
        words = raw_text.split()
        return " ".join(words[:100])
