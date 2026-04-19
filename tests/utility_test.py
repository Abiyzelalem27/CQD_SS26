

import numpy as np 
import pytest
from numpy.testing import assert_allclose

from Comp_Quant_Dynam import (
    I, X, Y, Z, eigenfunction, double_well,
)


def test_I_gate():
    """check that I is unitary"""
    assert np.allclose(I.conj().T @ I, I)

def test_sigmax():
    """check that X is unitary and self-inverse"""
    assert np.allclose(X.conj().T @ X, I)
    assert np.allclose(X @ X, I)

def test_sigmay():
    """check that Y is unitary and self-inverse"""
    assert np.allclose(Y.conj().T @ Y, I)
    assert np.allclose(Y @ Y, I)

def test_sigmaz():
    """check that Z is unitary and self-inverse"""
    assert np.allclose(Z.conj().T @ Z, I)
    assert np.allclose(Z @ Z, I)  

def test_eigenfunction_normalization():
    x_test = np.linspace(-5, 5, 200)
    psi0 = eigenfunction(0, x_test)
    norm = np.trapz(psi0**2, x_test)
    assert np.isclose(norm, 1.0, atol=1e-2), "Eigenfunction not normalized"

def test_double_well_symmetry():
    x_sym = np.linspace(-2, 2, 100)
    lam = 0.1
    V = double_well(x_sym, lam)
    sym_error = np.max(np.abs(V - V[::-1]))
    assert sym_error < 1e-12, "Potential is not symmetric"

def test_double_well_center():
    V0 = double_well(np.array([0.0]), 0.1)[0]
    assert np.isclose(V0, 0.0), "V(0) must be zero"  
