import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse.linalg import svds


class TruncatedSVD(BaseEstimator):
    def __init__(self, k=100):
        self.k = k
        self.U_k = None
        self.D_k = None
        self.VT_k = None

    def fit(self, X):
        A = X.T.astype(float)

        u, s, vt = svds(A, k=self.k)
        if u is None or s is None or vt is None:
            raise ValueError("SVD failed to compute. Check the input matrix.")

        idx = np.argsort(s)[::-1]

        self.U_k = u[:, idx]
        self.D_k = np.diag(s[idx])
        self.VT_k = vt[idx, :]

        return self

    def transform(self, X):
        return X @ self.U_k

    def fit_transform(self, X, y=None):
        self.fit(X)
        if self.D_k is not None and self.VT_k is not None:
            return (self.D_k @ self.VT_k).T
        else:
            raise ValueError(
                "SVD components not found. Ensure fit() was called successfully."
            )
