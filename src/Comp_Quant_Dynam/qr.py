

import numpy as np
from numpy.linalg import qr 

def power_iteration(A, n, tol):
    """
    Power Iteration Method: Computes the dominant eigenvector of a matrix A.
    
    Parameters
    A : ndarray(Input square matrix).
    n : int(Maximum number of iterations).
    tol : float(Convergence tolerance).

    Returns
    v : Approximation of the dominant eigenvector.
    """
    v = np.random.normal(size=A.shape[1])  # Random initial vector
    v = v / np.linalg.norm(v)              # Normalize initial vector
    previous = np.empty(shape=A.shape[1])       # Store previous iteration
    while True:
        previous[:] = v
        v = A @ v  # Matrix-vector multiplication
        v = v / np.linalg.norm(v) # Normalize vector
        if np.allclose(v, previous, atol=tol): # Check convergence
            break
    return v


def simultaneous_orthogonalization(A, n, tol):
    """
    Simultaneous Orthogonalization Method:, Computes multiple eigenvectors simultaneously using
    repeated orthogonalization via QR decomposition.

    where QR decomposition orthogonalizes the vectors at every iteration step.
    """
    Q, R = np.linalg.qr(A)  # Initial QR decomposition
    previous = np.empty(shape=Q.shape) # Store previous Q
    for i in range(n):
        previous[:] = Q
        X = A @ Q
        Q, R = np.linalg.qr(X)  # Orthogonalization
        if np.allclose(Q, previous, atol=tol):
            break
    return Q 
    

def qr_algorithm(A, n, tol):
    """
    QR Algorithm for Eigenvalue Computation, Computes eigenvalues/eigenvectors using repeated
    QR decompositions. The matrix gradually converges toward upper triangular (or diagonal for symmetric matrices) form, 
    where the diagonal entries are the eigenvalues.

    Returns
    -------
    Q : ndarray
        Approximate eigenvector matrix.
    """
    Q, R = np.linalg.qr(A) # Initial QR decomposition
    previous = np.empty(shape=Q.shape)  #iteration
    for i in range(n):
        previous[:] = Q
        X = R @ Q
        Q, R = np.linalg.qr(X) # QR decomposition
        if np.allclose(X, np.triu(X), atol=tol):  # Check if matrix becomes upper triangular
            break
    return Q