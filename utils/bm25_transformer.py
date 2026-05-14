import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.idf_ = None
        self.avg_dl_ = None

    def fit(self, X):
        n_samples, _ = X.shape

        df = np.array((X > 0).sum(axis=0)).flatten()
        self.idf_ = np.log((n_samples - df + 0.5) / (df + 0.5) + 1.0)

        self.avg_dl_ = X.sum(axis=1).mean()
        return self

    def transform(self, X):
        X = sp.csr_matrix(X)
        doc_lengths = np.array(X.sum(axis=1)).flatten()

        rows, cols = X.nonzero()
        data = X.data

        norm_factor = self.k1 * (
            1 - self.b + self.b * (doc_lengths[rows] / self.avg_dl_)
        )
        new_data = data * (self.k1 + 1) / (data + norm_factor)

        if self.idf_ is not None:
            new_data = new_data * self.idf_[cols]
            return sp.csr_matrix((new_data, (rows, cols)), shape=X.shape)
        else:
            raise ValueError(
                "The BM25Transformer must be fitted before calling transform()"
            )
