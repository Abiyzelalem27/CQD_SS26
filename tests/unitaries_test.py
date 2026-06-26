import numpy as np

import Comp_Quant_Dynam.unitaries as unit
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.hamiltonians as hams
import Comp_Quant_Dynam.operators as ops
import pytest


#################### Solution sheet 2 #####################

class Test_init_coeffs_eigenbasis:

    H = np.array([[0, 1], [1, 0]]) # simple \sigma_x Hamiltonian
    evals, evecs = np.linalg.eigh(H)

    def test_init_coeffs_eigenbasis_HO_ground(self):
        psi0 = np.array([1, -1]) * 1/np.sqrt(2) # ground state of \sigma_x
        expected = np.array([1, 0]) # in the eigenbasis of \sigma_x, the ground state
        result = unit.init_coeffs_eigenbasis(psi0, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the initial state is approximately the ground state in the eigenbasis, up to a global phase

    def test_init_coeffs_eigenbasis_HO_computational(self):
        psi0 = np.array([0, 1]) # state in the computational basis
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # in the eigenbasis of \sigma_x, this is an equal superposition of the two eigenvectors
        result = unit.init_coeffs_eigenbasis(psi0, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the initial state is approximately the expected state in the eigenbasis, up to a global phase

class Test_t_evol_eigenbasis:

    H = np.array([[0, 1], [1, 0]]) # simple \sigma_x Hamiltonian
    evals, evecs = np.linalg.eigh(H)

    def test_t_evol_eigenbasis_HO_ground(self):
        init_coeffs = np.array([1, 0]) # ground state in the eigenbasis
        t = np.pi / 2
        expected = np.array([1j, -1j]) * 1/np.sqrt(2) # after time evolution, this should be the state in the computational basis corresponding to the second eigenvector
        result = unit.t_evol_eigenbasis(init_coeffs, t, self.evals, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the evolved state

    def test_t_evol_eigenbasis_HO_computational(self):
        init_coeffs = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) # equal superposition in the eigenbasis
        t = np.pi / 2
        expected = np.array([1, 0]) # after time evolution, this should be the state in the computational basis.
        result = unit.t_evol_eigenbasis(init_coeffs, t, self.evals, self.evecs)
        overlap = np.vdot(expected, result)
        assert np.isclose(np.abs(overlap), 1) # check that the evolved state is approximately the expected state in the computational basis, up to a global phase

    
#################### Solution sheet 3 #####################


class Test_t_evol_split_step_fourier:

    L = 20
    npoints = 501
    xvals, dx = util.create_xvals(L, npoints)
    

    def test_t_evol_split_step_fourier_free_particle(self):
        x0 = 0
        sigma = 1
        p0 = 1
        tvec = util.create_tvecs(tsteps=100, dt=0.1)
        psi0 = util.gaussian_wave_packet(self.xvals, x0=x0, sigma=sigma, p0=p0) # Gaussian wave packet with some momentum
        V_func = lambda t: np.zeros_like(self.xvals) # free particle
        psit = unit.t_evol_split_step_fourier(psi0, V_func, tvec, self.xvals)
        # check that the norm is approximately conserved
        norms = np.sum(np.abs(psit)**2, axis=1) * self.dx
        assert np.allclose(norms, 1)

    def test_t_evol_split_step_fourier_harmonic_oscillator(self):
        x0 = -2
        sigma = 1/np.sqrt(2)
        p0 = 3
        tvec_2pi = util.create_tvecs(tsteps=200, dt=np.pi/100) # choose time steps to be small compared to the period of the harmonic oscillator
        psi0 = util.gaussian_wave_packet(self.xvals, x0=x0, sigma=sigma, p0=p0) #coherent state
        V_func = lambda t: 0.5 * self.xvals**2 # harmonic oscillator potential
        psit = unit.t_evol_split_step_fourier(psi0, V_func, tvec_2pi, self.xvals)
        overlap = np.vdot(psi0, psit[-1]) * self.dx
        assert np.isclose(np.abs(overlap), 1) # check that the state approximately returns to itself after one period of the harmonic oscillator, up to a global phase

    
#################### Exercise sheet 6 #####################

class Test_calc_expv_ED:
    
    def test_calc_expv_ED(self):
        N = 8
        state = np.eye(1, 2**N, 0).flatten() # initial state |000...0>

        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_z = ops.sigma_z_sparse() # single-site sigma_z operator

        local_dims = [2] * N
        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x)
        z3 = ops.n_party_op_sparse(local_dims, 3, sig_z)
        z4 = ops.n_party_op_sparse(local_dims, 4, sig_z)
        obsv_vec = np.array([z3, z4]) # we want to compute the expectation values of sigma_z on sites 3 and 4

        t_end = 2 * np.pi
        t_steps = 1000
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)
        expv = unit.calc_expv_ED(obsv_vec, H, state, tvec)
        assert expv.shape == (2, t_steps + 1) # check that the output has the correct shape
        assert np.all(np.abs(expv) <= 1) # check that the expectation values are between -1 and 1, as expected for sigma_z
        assert np.allclose(expv[1, :], -1) # check that the expectation value of sigma_z on site 4 is always -1, since the Hamiltonian only acts on site 3 and does not affect site 4
        expected = -1 * np.cos(tvec) # the expectation value of sigma_z on site 3 should oscillate as -cos(t) since the Hamiltonian is a sigma_x term on site 3
        diff = expv[0, :] - expected
        assert np.allclose(expv[0, :], expected, atol = 1e-2), (
            "Expectation value of sigma_z on site 3 does not match the expected oscillation; "
            f"max_abs_diff={np.max(np.abs(diff))}, "
            f"midpoint={expv[0, t_steps // 2]}, "
            f"expected_midpoint={expected[t_steps // 2]}"
        ) # check that the expectation value of sigma_z on site 3 approximately matches the expected oscillation, allowing for some numerical error

    def test_calc_expv_ED_warnings(self):
        N = 8
        state = np.eye(1, 2**N, 0).flatten() # initial state |000...0>

        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_z = ops.sigma_z_sparse() # single-site sigma_z operator
        
        non_hermitian_op = sig_x + 1j * sig_z # non-Hermitian operator
        nh3 = ops.n_party_op_sparse([2] * N, 3, non_hermitian_op) # non-Hermitian operator acting on site 3

        local_dims = [2] * N

        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x)
        
        t_end = 2 * np.pi
        t_steps = 100
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)
        pytest.warns(UserWarning, unit.calc_expv_ED, nh3, H, state, tvec)
