import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.neighbors import NearestNeighbors


def align(src, refs):
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(refs)
    _, indices = neighbors.kneighbors(src)
    return refs[indices.squeeze(), :]


def factorize(X, P, random_state=0):
    """
    Factorizes the aligned features X to find a content space of rank P.

    Args:
        X: aligned features of shape ``(T, K, N)``, where T is the number of frames, K is the number of speakers, and N is the feature dimension.
            Storing speakers along the second axis lets the block matrix ``(T, K * N)`` be formed without a copy.
        P: rank of the content space.
        random_state: seed for the randomized SVD.

    Returns:
        C: content space of shape ``(T, P)``
        M: average speaker transformation of shape ``(P, N)``
        S: speaker-specific transformations of shape ``(K, P, N)``
    """
    T, K, N = X.shape
    C, A, ST = randomized_svd(X.reshape(T, -1), n_components=P, random_state=random_state)
    C = C * np.sqrt(A)
    S = np.sqrt(A[:, None]) * ST
    S = S.reshape(P, K, N).transpose(1, 0, 2)
    M = np.mean(S, axis=0)
    return C, M, S
