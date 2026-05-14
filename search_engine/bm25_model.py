from abc import ABC
from search_engine.search_model import SearchModel
from config import SHARED_MODELS_BM25_FILE
from utils.preprocess_text import TextPreprocessor
from utils.vocab_file import load_vocab_from_file
import joblib


# Global class for BM25 search model.
class BM25SearchModel(SearchModel, ABC):
    def __init__(self):
        super().__init__()
        self.text_preprocessor = TextPreprocessor()
        self.vocab = load_vocab_from_file()
        shared_data = joblib.load(SHARED_MODELS_BM25_FILE)
        self.vectorizer = shared_data["vectorizer"]
        self.bm25 = shared_data["bm25"]
