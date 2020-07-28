import numpy as np
from sklearn.preprocessing import normalize


def pagerank(
    x: np.ndarray, df: float = 0.85, max_iter: int = 50, method: str = "iterative"
) -> np.ndarray:
    """
    PageRank method
    ==================
    
    Arguments
    ---------
    x : np.ndarray
    df : float
        Damping Factor, 0 < df < 1
    max_iter : int
        Maximum number of iteration for Power method
    method : str
        default is iterative, oter algebraic
        
    Returns
    -------
    R : np.ndarray
        PageRank vector (score)
    """

    assert 0 < df < 1

    A = normalize(x, axis=0, norm="l1")
    N = np.ones(A.shape[0]) / A.shape[0]

    if method == "iterative":
        R = np.ones(A.shape[0])
        # iteration
        for _ in range(max_iter):
            R = df * np.matmul(A, R) + (1 - df) * N
    elif method == "algebraic":
        I = np.eye(A.shape[0])
        R = np.linalg.inv((I - df * A))
        R = np.matmul(R, (1 - df) * N)

    return R
