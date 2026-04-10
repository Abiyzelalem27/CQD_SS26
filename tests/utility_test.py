

import numpy as np 
import pytest
from numpy.testing import assert_allclose

from Comp_Quant_Dynam import (
    I, X, Y, Z,
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
