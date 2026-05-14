import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.idf_ = None

    def fit(self, X):
        N, _ = X.shape

        df = np.array((X > 0).sum(axis=0)).flatten().astype(float)

        self.idf_ = np.log(N / (df + 1e-9))
        return self

    def transform(self, X):
        X = sp.csr_matrix(X).copy()
        rows, cols = X.nonzero()

        if self.idf_ is not None:
            X.data = X.data * self.idf_[cols]
        else:
            raise ValueError(
                "The TfidfTransformer must be fitted before calling transform()"
            )

        norms = np.sqrt(np.array(X.multiply(X).sum(axis=1)).flatten())
        norms[norms == 0] = 1.0
        X.data /= norms[rows]

        return X
