from pathlib import Path
import os

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_final"
os.makedirs(DATA_DIR, exist_ok=True)
VOCAB_FILE = DATA_DIR / "golden_vocabulary_final_2000"

# TODO: change for the final parmeters
# Config for golden Vocabulary
FLUSH_INTERVAL = 50000
PRINT_INTERVAL = 10000
MIN_DF_FLUSH_THRESHOLD = 1

MIN_DF = 5
MAX_DF_RATIO = lambda total_chunks: int(total_chunks * 0.50)
MAX_FEATURES = 100000

# Config for Database
DB_PATH = DATA_DIR / "gutenberg_2000.db"

# Config for chromaDB
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
OLLAMA_EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "book_chunks"

# Config for Scrapper
NO_CHUNK_TO_SAVE = 500
NO_BOOKS = 6000
MAX_NO_OF_CHUNKS = 250000
NO_WORDS_PER_CHUNK = 200
OVERLAP_SIZE = 10

# Config for SVD
SVD_COMPONENTS = [5, 25, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]

TFIDF_DIR = DATA_DIR / "tfidf"
os.makedirs(TFIDF_DIR, exist_ok=True)

SHARED_MODELS_TFIDF_FILE = TFIDF_DIR / "shared_models_2000.joblib"
RAW_MATRIX_TFIDF = TFIDF_DIR / "matrix_raw_2000.joblib"


# Config for BM25
BM25_DIR = DATA_DIR / "bm25"
os.makedirs(BM25_DIR, exist_ok=True)
SHARED_MODELS_BM25_FILE = BM25_DIR / "shared_models_bm25_2000.joblib"
RAW_MATRIX_BM25 = BM25_DIR / "matrix_bm25_2000.joblib"
