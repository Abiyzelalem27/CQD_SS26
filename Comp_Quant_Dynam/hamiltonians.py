<<<<<<< HEAD
import numpy as np   # standard numerics library
# standard numerics 
import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite as herm
from scipy.special import factorial
=======


# standard numerics 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
>>>>>>> 32c83e9ff15b970e77206e6886ad427876d2a2f4
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
<<<<<<< HEAD
    ------- 
    array_like
        Values of the normalized eigenfunction ψ_n(x).
    """
    Hn = herm(n)
=======
    -------
    array_like
        Values of the normalized eigenfunction ψ_n(x).
    """
    Hn = hermite(n)
>>>>>>> 32c83e9ff15b970e77206e6886ad427876d2a2f4
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

<<<<<<< HEAD


def HO_eigenstates_exact(n,x):
    """
    Returns the n-th eigenstate of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    normalization = 1 / np.sqrt(2 ** n * math.factorial(n) * np.sqrt(np.pi)) 
    return normalization * herm(n)(x) * np.exp(-x ** 2 / 2)


def HO_eigenenergies_exact(n):
    """
    Returns the n-th eigenenergy of the quantum harmonic oscillator in numerical units.
    """

    return n + 0.5

def H_kinetic(x):
    """
    Returns the kinetic energy operator of the quantum harmonic oscillator in the position basis for a grid 'x'.
    The kinetic energy operator is represented as a finite difference matrix approximating the second derivative, which is given by the formula:
    T = - (ħ^2 / 2m) * d^2/dx^2, or in numerical units, T = -0.5 * d^2/dx^2. The second derivative can be approximated using the central difference formula:
    d^2ψ/dx^2 ≈ (ψ(x + dx) - 2ψ(x) + ψ(x - dx)) / (dx^2).
    # Kinetic energy term in discrete form 
    # finite difference approximation
    # i+1 → upper diagonal
    # i-1 → lower diagonal
    # -2i → main diagonal
    """

    n_points = len(x) # number of grid points
    dx = x[1] - x[0] # grid spacing
    main_diag = np.diag(np.ones(n_points))
    off_diag = -0.5 * np.diag(np.ones(n_points - 1), k=1)
    return (main_diag + off_diag + off_diag.T) / (dx * dx)

def HO_potential(x):
    """
    Returns the potential energy operator of the quantum harmonic oscillator in the position basis for a grid 'x'.
    The potential energy operator is represented as a diagonal matrix with elements given by V(x) = 0.5 * x^2.
    """
    
    return 0.5 * np.diag(x ** 2) 

=======
>>>>>>> 32c83e9ff15b970e77206e6886ad427876d2a2f4
