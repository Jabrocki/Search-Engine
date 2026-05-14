import joblib
from sklearn.feature_extraction import DictVectorizer
from db.models import db, Chunk
from utils.vocab_file import load_vocab_from_file
from config import SVD_COMPONENTS, RAW_MATRIX_BM25, SHARED_MODELS_BM25_FILE, BM25_DIR
from utils.bm25_transformer import BM25Transformer
from utils.truncated_svd import TruncatedSVD


def build_and_save_everything_bm25():
    vocab = load_vocab_from_file()
    chunk_ids = []

    def stream_filtered_chunks():
        query = Chunk.select(Chunk.id, Chunk.compressed_json).iterator()
        iter = 0
        for chunk in query:
            chunk_ids.append(chunk.id)
            bow_dict = chunk.get_json()
            if iter % 10000 == 0:
                print(f"Processed {iter} chunks for BM25...")
            iter += 1
            yield {w: c for w, c in bow_dict.items() if w in vocab}

    vectorizer = DictVectorizer(sparse=True)
    X_counts = vectorizer.fit_transform(stream_filtered_chunks())

    bm25 = BM25Transformer()
    X_bm25 = bm25.fit_transform(X_counts)

    print("Saving shared_models_bm25.joblib...")
    shared_models = {"vectorizer": vectorizer, "bm25": bm25}
    joblib.dump(shared_models, SHARED_MODELS_BM25_FILE)

    print("Saving matrix_bm25.joblib...")
    raw_data = {"index_matrix": X_bm25, "chunk_ids": chunk_ids}
    joblib.dump(raw_data, RAW_MATRIX_BM25)

    del X_counts

    for n in SVD_COMPONENTS:
        print(f"Computing TruncatedSVD for n={n}...")
        svd = TruncatedSVD(k=n)
        X_svd_dense = svd.fit_transform(X_bm25)

        joblib.dump(
            {
                "index_matrix": X_svd_dense,
                "chunk_ids": chunk_ids,
                "svd_model": svd,
            },
            BM25_DIR / f"bm25_svd_{n}.joblib",
        )

    print("All BM25 matrices computed and saved.")


if __name__ == "__main__":
    db.connect()
    build_and_save_everything_bm25()
    db.close()
