import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def cosine_similarity(query_vector, document_matrix):
    dot_product = query_vector @ document_matrix.T

    if sp.issparse(dot_product):
        dot_product = dot_product.toarray()
    dot_product = dot_product.flatten()

    if sp.issparse(query_vector):
        norm_q = spla.norm(query_vector)
    else:
        norm_q = np.linalg.norm(query_vector)

    if sp.issparse(document_matrix):
        norm_docs = np.sqrt(np.array(document_matrix.power(2).sum(axis=1))).flatten()
    else:
        norm_docs = np.linalg.norm(document_matrix, axis=1)

    norm_q = norm_q if norm_q > 0 else 1.0
    norm_docs[norm_docs == 0] = 1.0

    similarities = dot_product / (norm_q * norm_docs)
    return np.abs(similarities)
