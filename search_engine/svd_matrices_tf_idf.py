import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from db_models import db, Chunk
from utils import load_vocabulary


def build_and_save_everything_tfidf():
    vocab = load_vocabulary()
    chunk_ids = []

    print("Stage 1: Building shared TF-IDF model...")

    def stream_filtered_chunks():
        query = Chunk.select(Chunk.id, Chunk.compressed_json).iterator()
        iter = 0
        for chunk in query:
            chunk_ids.append(chunk.id)
            bow_dict = chunk.get_data()
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
    joblib.dump(shared_models, "shared_models.joblib")

    print("Saving matrix_raw.joblib...")
    raw_data = {"index_matrix": X_tfidf, "chunk_ids": chunk_ids}
    joblib.dump(raw_data, "matrix_raw.joblib")

    del X_counts

    print("Stage 2: Calculating SVD matrices...")
    n_components_list = [25, 50, 100, 200, 300, 400]

    for n in n_components_list:
        print(f"Computing TruncatedSVD for n={n}...")
        svd = TruncatedSVD(n_components=n)
        X_svd_dense = svd.fit_transform(X_tfidf)

        filename = f"matrix_svd_{n}.joblib"
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
