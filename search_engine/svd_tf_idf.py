import joblib
from sklearn.feature_extraction import DictVectorizer
from utils.tfidf_transformer import TfidfTransformer
from utils.truncated_svd import TruncatedSVD
from db.models import db, Chunk
from utils.vocab_file import load_vocab_from_file
from config import SVD_COMPONENTS, RAW_MATRIX_TFIDF, SHARED_MODELS_TFIDF_FILE, TFIDF_DIR


def build_and_save_everything_tfidf():
    vocab = load_vocab_from_file()
    chunk_ids = []

    print("Stage 1: Building shared TF-IDF model...")

    def stream_filtered_chunks():
        query = Chunk.select(Chunk.id, Chunk.compressed_json).iterator()
        iter = 0
        for chunk in query:
            chunk_ids.append(chunk.id)
            bow_dict = chunk.get_json()
            if iter % 10000 == 0:
                print(f"Processed {iter} chunks for TF-IDF...")
            iter += 1
            yield {w: c for w, c in bow_dict.items() if w in vocab}

    vectorizer = DictVectorizer(sparse=True)
    X_counts = vectorizer.fit_transform(stream_filtered_chunks())

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)

    print("Saving shared_models.joblib...")
    shared_models = {"vectorizer": vectorizer, "tfidf": tfidf}
    joblib.dump(shared_models, SHARED_MODELS_TFIDF_FILE)

    print("Saving matrix_raw.joblib...")
    raw_data = {"index_matrix": X_tfidf, "chunk_ids": chunk_ids}
    joblib.dump(raw_data, RAW_MATRIX_TFIDF)

    del X_counts

    print("Stage 2: Calculating SVD matrices...")
    n_components_list = SVD_COMPONENTS

    for n in n_components_list:
        print(f"Computing TruncatedSVD for n={n}...")
        svd = TruncatedSVD(k=n)
        X_svd_dense = svd.fit_transform(X_tfidf)

        filename = TFIDF_DIR / f"tfidf_svd_{n}.joblib"
        print(f"Saving {filename}...")
        svd_bundle = {
            "index_matrix": X_svd_dense,
            "chunk_ids": chunk_ids,
            "svd_model": svd,
        }
        joblib.dump(svd_bundle, filename)

        del X_svd_dense
        del svd

    print("Process complete. All files saved.")


if __name__ == "__main__":
    db.connect()
    build_and_save_everything_tfidf()
    db.close()
