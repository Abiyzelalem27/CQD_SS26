import numpy as np

import pytest
import numpy as np
import numpy.linalg as LA
from numpy.testing import assert_allclose
from Comp_Quant_Dynam import I, X, Y, Z  

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

def test_double_well_symmetry():
    x_sym = np.linspace(-2, 2, 100)
    lam = 0.1
    V = double_well(x_sym, lam)
    sym_error = np.max(np.abs(V - V[::-1]))
    assert sym_error < 1e-12, "Potential is not symmetric"

def test_double_well_center():
    V0 = double_well(np.array([0.0]), 0.1)[0]
    assert np.isclose(V0, 0.0), "V(0) must be zero"  


class Test_HO_eigenstates_exact:

    L = 10
    npoints = 101
    xvals = np.linspace(-L / 2, L / 2, npoints)
    
    def test_HO_ground(self):
        n0 = 0
        expected =  1 / np.pi ** (1 / 4) * np.exp(-self.xvals ** 2 / 2)
        result = ham.HO_eigenstates_exact(n0, self.xvals)
        assert np.allclose(expected, result)

class Test_HO_eigenenergies:
    
    L = 15
    npoints = 401
    acc = 1e-2
    #npoints = 2001
    #acc = 1e-3
    xvals = np.linspace(-L / 2, L / 2, npoints)

    def test_HO_ED(self):
        
        H_pot = ham.HO_potential(self.xvals)
        H_kin = ham.H_kinetic(self.xvals)

        H_mat = H_pot + H_kin

        evals_num, evecs_num = LA.eigh(H_mat)
        evals_exact = ham.HO_eigenenergies_exact(np.arange(evals_num.size))
        assert np.allclose(evals_num[:10], evals_exact[:10], atol=self.acc)



        