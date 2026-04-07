import numpy as np   # standard numerics library
import math

def example_func(x):
    """
    Example function to demonstrate the repository structure.
    Returns the ground state wavefunction of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    return 1 / np.pi ** (1 / 4) * np.exp(-x ** 2 / 2)
