from peewee import SqliteDatabase, Model, IntegerField, TextField, BlobField
import zlib
import json

db = SqliteDatabase("/Volumes/External/gutenberg.db")


class Book(Model):
    id = IntegerField(unique=True)
    title = TextField()
    author = TextField(null=True)
    image_url = TextField(null=True)
    book_url = TextField(null=True)

    class Meta:
        database = db


class Chunk(Model):
    id = IntegerField(unique=True)
    book_id = IntegerField()
    chunk_url = TextField()
    compressed_json = BlobField()
    raw_text_blob = BlobField()

    class Meta:
        database = db

    def set_data(self, data_dict: dict):
        json_data = json.dumps(data_dict).encode("utf-8")
        self.compressed_json = zlib.compress(json_data)

    def get_data(self) -> dict:
        raw_data = bytes(self.compressed_json)
        decompressed = zlib.decompress(raw_data).decode("utf-8")
        return json.loads(decompressed)
