

# standard numerics 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from numpy.linalg import eigh 

sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])

def eigenfunction(n, x):
    """Compute the n-th eigenfunction of the quantum harmonic oscillator.

    The function evaluates the normalized wavefunction:
        ψ_n(x) = N_n H_n(x) exp(-x^2 / 2)

    where H_n is the physicist's Hermite polynomial of order n,
    and N_n is the normalization constant ensuring unit probability.

    Parameters
    ----------
    n : int
        Quantum number (energy level index).
    x : array_like
        Position values at which to evaluate the eigenfunction.

    Returns
    -------
    array_like
        Values of the normalized eigenfunction ψ_n(x).
    """
    Hn = hermite(n)
    norm = 1.0 / np.sqrt((2**n) * factorial(n) * np.sqrt(np.pi))
    return norm * Hn(x) * np.exp(-x**2 / 2) 

def double_well(x, lam):
    """function defines the potential energy V(x) of a
    quantum particle in a double-well system.

    Mathematical form:
        V(x) = -1/2 * x^2 + λ * x^4

    Explanation:
    - The term (-1/2 * x^2) creates TWO wells (a double shape)
    - The term (λ * x^4) controls the barrier height between them

    - Small λ  → shallow barrier (easy tunneling)
    - Large λ  → high barrier (wells become separated)

    Parameters:
    x   : position array (grid points in space)
    lam : strength of the quartic term (controls barrier)

    Returns:
    V(x): potential energy evaluated at each x
    """ 
    return -0.5 * x**2 + lam * x**4

