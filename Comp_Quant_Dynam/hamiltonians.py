import numpy as np   # standard numerics library
import math
from scipy.special import hermite as herm


#################### Solution sheet 1 ####################


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