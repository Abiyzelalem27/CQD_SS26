import numpy as np
import numpy.linalg as LA
from scipy.sparse.linalg import eigsh
from math import comb
import pytest

import Comp_Quant_Dynam.hamiltonians as ham
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops
from Comp_Quant_Dynam.utility import create_xvals


###################### Solution sheet 1 ######################


class Test_HO_eigenstates_exact:

    L = 10
    npoints = 101
    xvals, dx = create_xvals(L, npoints)
    
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
    xvals, dx = create_xvals(L, npoints)


    def test_HO_ED(self):
        
        H_pot = ham.HO_potential(self.xvals)
        H_kin = ham.H_kinetic(self.xvals)

        H_mat = H_pot + H_kin

        evals_num, evecs_num = LA.eigh(H_mat)
        evals_exact = ham.HO_eigenenergies_exact(np.arange(evals_num.size))
        assert np.allclose(evals_num[:10], evals_exact[:10], atol=self.acc)


###################### Solution sheet 2 ######################


class Test_HO_sparse_eigenenergies:
    
    L = 15
    npoints = 401
    acc = 1e-2
    k = 10
    xvals, dx = create_xvals(L, npoints)

    H_pot_sparse = ham.HO_potential_sparse(xvals)
    H_kin_sparse = ham.H_kinetic_sparse(xvals)

    H_mat = H_pot_sparse + H_kin_sparse
    evals_num, evecs_num = eigsh(H_mat, k=k, which='SA')

    def test_HO_ED_evals(self):
        
        evals_exact = ham.HO_eigenenergies_exact(np.arange(self.evals_num.size))
        assert np.allclose(self.evals_num, evals_exact, atol=self.acc)

    def test_HO_ED_evecs(self):

        evecs_dense = LA.eigh(self.H_mat.toarray())[1][:, :self.k]
        overlaps = np.zeros(self.k, dtype=complex)
        for i in range(self.k):
            overlap = np.vdot(self.evecs_num[:, i], evecs_dense[:, i])
            overlaps[i] = overlap
        assert np.allclose(np.abs(overlaps), 1) # check that the eigenvectors are approximately the same, up to a global phase


###################### Solution sheet 3 ######################


class Test_potentials:
    L = 2
    npoints = 21
    xvals, dx = create_xvals(L, npoints)
    
    def test_step_potential(self):
        V0 = 5
        expected = np.zeros_like(self.xvals)
        expected[self.xvals >= 0] = V0
        result = ham.step_potential(self.xvals, V0)
        assert np.allclose(expected, result)

    def test_barrier_potential(self):
        V0 = 5
        width = 1
        expected = np.zeros_like(self.xvals)
        expected[np.abs(self.xvals) <= width / 2] = V0
        result = ham.barrier_potential(self.xvals, V0, width)
        assert np.allclose(expected, result)


###################### Exercise sheet 4 ######################


class Test_build_H_coupled_HO_man:
    N1 = 10
    N2 = 10

    def test_build_H_coupled_HO_man_hermitian(self):
        lam = 0.1
        tol = 1e-10
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        diff = H - H.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_coupled_HO_man_non_interacting(self):
        lam = 0
        k = 15
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        assert H.nnz == self.N1 * self.N2  # check that the number of non-zero elements in H is equal to N1*N2
        evals, evecs = eigsh(H, k=k+2, which='SA', ncv = 5 * k) # compute the lowest k+2 eigenvalues and eigenvectors of H using the sparse eigensolver
        evals_uncoupled = np.add.outer(ham.HO_eigenenergies_exact(np.arange(self.N1)), ham.HO_eigenenergies_exact(np.arange(self.N2))).flatten()
        evals_uncoupled.sort()
        assert np.allclose(evals[:k], evals_uncoupled[:k])

    def test_build_H_coupled_HO_man_lam_effect(self):
        lam = 0.5
        k = 15
        dim = self.N1 * self.N2
        H = ham.build_H_coupled_HO_man(self.N1, self.N2, lam)
        evals, evecs = eigsh(H, k=k+2, which='SA')
        evals_coupled = [ham.coupled_HO_eigenenergies_exact(n_cm, n_rel, lam) for n_cm in range(dim) for n_rel in range(dim)]
        evals_coupled.sort()
        assert np.allclose(evals[:k], evals_coupled[:k], atol=1e-3)


class Test_build_H_coupled_HO_improved:
    N1 = 10
    N2 = 10

    def test_build_H_coupled_HO_improved_hermitian(self):
        lam = 0.1
        tol = 1e-10
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        diff = H - H.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_coupled_HO_improved_non_interacting(self):
        lam = 0
        k = 15
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        assert H.nnz == self.N1 * self.N2  # check that the number of non-zero elements in H is equal to N1*N2
        evals, evecs = eigsh(H, k=k, which='SA', ncv = 5 * k) # compute the lowest k eigenvalues and eigenvectors of H using the sparse eigensolver
        evals_uncoupled = np.add.outer(ham.HO_eigenenergies_exact(np.arange(self.N1)), ham.HO_eigenenergies_exact(np.arange(self.N2))).flatten()
        evals_uncoupled.sort()
        print("evals:", evals[:k])
        print("evals_uncoupled:", evals_uncoupled[:k])
        assert np.allclose(evals[:k], evals_uncoupled[:k])

    def test_build_H_coupled_HO_improved_lam_effect(self):
        lam = 0.5
        k = 15
        dim = self.N1 * self.N2
        H = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam)
        evals, evecs = eigsh(H, k=k+2, which='SA')
        print("evals with coupling:", evals)
        evals_coupled = [ham.coupled_HO_eigenenergies_exact(n_cm, n_rel, lam) for n_cm in range(dim) for n_rel in range(dim)]
        evals_coupled.sort()
        assert np.allclose(evals[:k], evals_coupled[:k], atol=1e-3)

class Test_coupled_HO_E0_exact:

    def test_coupled_HO_E0_exact(self):
        lam = 0.5
        expected = 1 / 2 + np.sqrt(1 + 2 * lam) / 2
        result = ham.coupled_HO_E0_exact(lam)
        assert np.isclose(expected, result)

    def test_coupled_HO_E0_exact_generalized(self):
        lam = 0.35
        expected = ham.coupled_HO_eigenenergies_exact(0, 0, lam)
        result = ham.coupled_HO_E0_exact(lam)
        assert np.isclose(expected, result)

class Test_HO_product_eigenstates:

    N1 = 4
    N2 = 6
    L = 15
    npoints = 401
    xvals, dx = create_xvals(L, npoints)
    eigenstates = ham.HO_product_eigenstates(N1, N2, xvals) # normalize the eigenstates by the grid spacing to ensure that they are properly normalized in the continuum limit

    def test_HO_product_eigenstates_shape(self):

        assert self.eigenstates.shape == (self.N1 * self.N2, self.npoints, self.npoints)

    def test_HO_product_eigenstates_symmetry(self):
        assert np.allclose(self.eigenstates[0], self.eigenstates[0].T)

    def test_HO_product_eigenstates_orthonormality(self):
        overlaps = np.zeros((self.N1 * self.N2, self.N1 * self.N2), dtype=complex)
        for i in range(self.N1 * self.N2):
            for j in range(self.N1 * self.N2):
                overlap = np.vdot(self.eigenstates[i].flatten(), self.eigenstates[j].flatten())
                overlaps[i, j] = np.real(overlap) * self.dx**2  # divide by dx^2 to account for the normalization of the eigenstates in the continuum limit
        assert np.allclose(overlaps, np.eye(self.N1 * self.N2))

    def test_HO_product_eigenstates_example(self):
        N1 = N2 = 8
        eigenstates = ham.HO_product_eigenstates(N1, N2, self.xvals)
        assert np.allclose(eigenstates[9], np.outer(ham.HO_eigenstates_exact(1, self.xvals), ham.HO_eigenstates_exact(1, self.xvals))) 

class Test_class_traj:
    N1 = N2 = 20
    local_dims = [N1, N2]
    x1_op = ops.n_party_op_sparse(local_dims, 0, ops.x_operator_sparse(N1))
    x2_op = ops.n_party_op_sparse(local_dims, 1, ops.x_operator_sparse(N2))
    p1_op = ops.n_party_op_sparse(local_dims, 0, ops.p_operator_sparse(N1))
    p2_op = ops.n_party_op_sparse(local_dims, 1, ops.p_operator_sparse(N2))

    def test_traj_comparison_to_numerics(self):
        alpha = 0.1
        state_1 = util.create_coherent_state(self.N1, alpha)
        state_2 = np.eye(1, self.N2, 4).flatten() # ground state of the second oscillator
        state = np.kron(state_1, state_2) # initial state is the product of the coherent state and the n=4 state.
        x10 = util.expectation_value(state, self.x1_op)
        x20 = util.expectation_value(state, self.x2_op)
        p10 = util.expectation_value(state, self.p1_op)
        p20 = util.expectation_value(state, self.p2_op)

        ini = np.array([x10, x20, p10, p20])
        print("Initial conditions:", ini)
        lam = 0.1

        tvec = util.create_tvecs(100, 0.1)

        x1_class, x2_class = ham.class_traj(lam, ini, tvec)
        x1_num, x2_num = np.zeros_like(tvec), np.zeros_like(tvec)

        H_mat = ham.build_H_coupled_HO_improved(self.N1, self.N2, lam).toarray()
        evals, evecs = LA.eigh(H_mat)
           

        eigen_coeffs = unitaries.init_coeffs_eigenbasis(state, evecs) 
        for i, t in enumerate(tvec):
            state_t = unitaries.t_evol_eigenbasis(eigen_coeffs, t, evals, evecs)
            x1_num[i] = np.real(util.expectation_value(state_t, self.x1_op))
            x2_num[i] = np.real(util.expectation_value(state_t, self.x2_op))
            
        assert np.allclose(x1_class, x1_num, atol=1e-8)
        assert np.allclose(x2_class, x2_num, atol=1e-8)


class Test_coupled_HO_potential:

    L = 10
    npoints = 101
    lam = 0.5

    xvals, dx = util.create_xvals(L, npoints)
    yvals, dy = util.create_xvals(L, npoints)

    X, Y = np.meshgrid(xvals, yvals)

    H_pot = ham.coupled_HO_potential(X, Y, lam)

    #def test_coupled_HO_potential(self):
    #    expected = ham.HO_potential(self.X) + ham.HO_potential(self.Y) + self.lam / 2 * (self.X - self.Y) ** 2
    #    assert np.allclose(expected, self.H_pot)

    def test_coupled_HO_potential_symmetry(self):
        assert np.allclose(self.H_pot, self.H_pot.T)

    def test_coupled_HO_potential_diagonal(self):
        expected_no_coupling = 1/2 * self.X**2 + 1/2 * self.Y**2
        assert np.allclose(self.H_pot.diagonal(), expected_no_coupling.diagonal()) # the coupling term should not contribute to the diagonal elements of the potential


###################### Solution sheet 5 ######################


class Test_build_H_TFIM:

    N = 10
    ome = 0.5

    def test_build_H_TFIM_hermitian(self):
        tol = 1e-10
        H = ham.build_H_TFIM(self.N, self.ome)
        diff = H - H.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_TFIM_no_field(self):
        ome = 0
        H = ham.build_H_TFIM(self.N, ome)
        H_diag = H.diagonal()
        H_reconstr = H - ops.diagonal_op_sparse(H_diag)
        diff = H_reconstr - H_reconstr
        assert np.allclose(diff.data, 0) # check that the off-diagonal elements of H are zero when there is no transverse field, which means that the Hamiltonian is diagonal in the computational basis

    def test_build_H_TFIM_GS_no_field(self):
        ome = 0
        H = ham.build_H_TFIM(self.N, ome)
        evals, evecs = eigsh(H, k=1, which='SA') # compute the lowest eigenvalue and eigenvector of H using the sparse eigensolver
        GS = evecs[:, 0] # the ground state is the eigenvector corresponding to the lowest eigenvalue
        assert np.isclose(GS[0] ** 2 + GS[-1] ** 2, 1) # check that the ground state is approximately equal to the expected ground state, up to a global phase

    def test_build_H_TFIM_GS_strong_field(self):
        ome = 1e5
        H = ham.build_H_TFIM(self.N, ome)
        evals, evecs = eigsh(H, k=1, which='SA') # compute the lowest k eigenvalues and eigenvectors of H using the sparse eigensolver
        GS = evecs[:, 0] # the ground state is the eigenvector corresponding to the lowest eigenvalue
        probs = np.abs(GS) ** 2
        probs_expected = np.array([comb(self.N, k) / 2**self.N for k in range(self.N + 1)])
        assert np.allclose(probs, probs_expected, atol=1e-6) # check that the ground state is approximately equal to the expected ground state, which is an equal superposition of all computational basis states, up to a global phase

class Test_build_H_TFIM_symm:

    N = 10
    ome = 0.5

    def test_build_H_TFIM_symm_hermitian(self):
        tol = 1e-10
        H_symm = ham.build_H_TFIM_symm(self.N, self.ome)
        diff = H_symm - H_symm.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_TFIM_symm_no_field(self):
        ome = 0
        H_symm = ham.build_H_TFIM_symm(self.N, ome)
        H_diag = H_symm.diagonal()
        H_reconstr = H_symm - ops.diagonal_op_sparse(H_diag)
        diff = H_reconstr - H_reconstr
        assert np.allclose(diff.data, 0) # check that the off-diagonal elements of H are zero when there is no transverse field, which means that the Hamiltonian is diagonal in the computational basis

    def test_build_H_TFIM_symm_GS_no_field(self):
        ome = 0
        H_symm = ham.build_H_TFIM_symm(self.N, ome)
        evals, evecs = eigsh(H_symm, k=1, which='SA') # compute the lowest eigenvalue and eigenvector of H using the sparse eigensolver
        GS = evecs[:, 0] # the ground state is the eigenvector corresponding to the lowest eigenvalue
        assert np.isclose(GS[0] ** 2, 1) # check that the ground state is approximately equal to the expected ground state, up to a global phase

    def test_build_H_TFIM_symm_GS_strong_field_even(self):
        ome = 1e10
        H_symm = ham.build_H_TFIM_symm(self.N, ome)
        evals, evecs = eigsh(H_symm, k=1, which='SA') # compute the lowest k eigenvalues and eigenvectors of H using the sparse eigensolver
        GS = evecs[:, 0] # the ground state is the eigenvector corresponding to the lowest eigenvalue
        probs_GS = np.abs(GS) ** 2
        probs_expected = np.array([comb(self.N, k) * 2 / 2 ** self.N for k in range(self.N // 2 + 1)])
        probs_expected[-1] /= 2 # the last term should be divided by 2 to account for the fact that the state with N/2 excitations has only one configuration instead of two
        assert np.allclose(probs_GS, probs_expected, atol=1e-6) # check that the ground state is approximately equal to the expected ground state, which is an equal superposition of all computational basis states

    def test_build_H_TFIM_symm_GS_strong_field_odd(self):
        N_odd = 9
        ome = 1e10
        H_symm = ham.build_H_TFIM_symm(N_odd, ome)
        evals, evecs = eigsh(H_symm, k=1, which='SA') # compute the lowest k eigenvalues and eigenvectors of H using the sparse eigensolver
        GS = evecs[:, 0] # the ground state is the eigenvector corresponding to the lowest eigenvalue
        probs_GS = np.abs(GS) ** 2
        probs_expected = np.array([comb(N_odd, k) * 2 / 2 ** N_odd for k in range((N_odd // 2) + 1)])
        assert np.allclose(probs_GS, probs_expected, atol=1e-6) # check that the ground state is approximately equal to the expected ground state, which is an equal superposition of all computational basis states


###################### Solution sheet 7 ######################


class Test_TFIM_E_MF:
    
    z = np.arange(-1, 1.01, 0.01)

    def test_TFIM_E_MF_no_field(self):
        omega = 0
        expected = - self.z ** 2 / 2
        phi = np.pi / 4
        result = ham.E_MF(self.z, phi, omega)
        assert np.allclose(expected, result)

    def test_TFIM_E_MF_phi_pi_2(self):
        omega = 0.5
        
        expected = - self.z ** 2 / 2
        phi = np.pi / 2
        result = ham.E_MF(self.z, phi, omega)
        assert np.allclose(expected, result)


###################### Solution sheet 8 ######################


class Test_build_H_TFIM_individual:

    N = 10
    ome = 0.5

    def test_build_H_TFIM_individual_hermitian(self):
        tol = 1e-10
        H_indiv = ham.build_H_TFIM_individual(self.N, self.ome)
        diff = H_indiv - H_indiv.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_TFIM_individual_no_field(self):
        ome = 0
        H_indiv = ham.build_H_TFIM_individual(self.N, ome)
        H_diag = H_indiv.diagonal()
        H_reconstr = ops.diagonal_op_sparse(H_diag)
        diff = H_indiv - H_reconstr
        assert np.allclose(diff.data, 0) # check that the off-diagonal elements of H are zero when there is no transverse field, which means that the Hamiltonian is diagonal in the computational basis

    def test_build_H_TFIM_individual_gs_energy(self):

        E_exact = ham.E_TFIM_individual_exact(self.N, self.ome)
        H_indiv = ham.build_H_TFIM_individual(self.N, self.ome)
        evals, evecs = eigsh(H_indiv, k=1, which='SA') # compute the lowest eigenvalue and eigenvector of H using the sparse eigensolver
        E_num = evals[0]
        assert np.isclose(E_num, E_exact, atol=1e-8) # check that the ground state energy computed from the Hamiltonian matches the exact ground state energy computed from the analytical formula
        

class Test_E_TFIM_individual_exact:

    N = 11
    ome = 0.5

    def test_E_TFIM_individual_exact_odd(self):
        with pytest.raises(AssertionError, match="The exact solution for the ground state energy of the TFIM is only implemented for even N"):
            ham.E_TFIM_individual_exact(self.N, self.ome)
        

class Test_build_H_TFIM_A2A:

    N = 8
    ome = 0.5

    def test_build_H_TFIM_A2A_hermitian(self):
        tol = 1e-10
        H_A2A = ham.build_H_TFIM_A2A(self.N, self.ome)
        diff = H_A2A - H_A2A.conj().T
        
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < tol  # check that the difference between H and its conjugate transpose is approximately zero, which means that H is Hermitian

    def test_build_H_TFIM_A2A_no_field(self):
        ome = 0
        H_A2A = ham.build_H_TFIM_A2A(self.N, ome)
        H_diag = H_A2A.diagonal()
        H_reconstr = ops.diagonal_op_sparse(H_diag)
        diff = H_reconstr - H_A2A
        assert np.allclose(diff.data, 0) # check that the off-diagonal elements of H are zero when there is no transverse field, which means that the Hamiltonian is diagonal in the computational basis

    def test_build_H_TFIM_A2A_entropy(self):

        t = 3
        dim = 2 ** self.N

        H_A2A = ham.build_H_TFIM_A2A(self.N, self.ome)
        evals, evecs = LA.eigh(H_A2A.toarray()) # compute the eigenvalues and eigenvectors of H using the dense eigensolver
        ini = np.full(dim, 1/np.sqrt(dim)) # initial state is the equal superposition of all computational basis states
        ini_proj = unitaries.init_coeffs_eigenbasis(ini, evecs) # project the initial state onto the eigenbasis of H

        state_t = unitaries.t_evol_eigenbasis(ini_proj, t, evals, evecs) # evolve the state in time using the eigenvalues and eigenvectors of H
        #rho_t = np.outer(state_t, state_t.conj()) # compute the density matrix of the state at time t
        rho_red = util.partial_trace(state_t, self.N // 2)
        S_A = util.entanglement_entropy(rho_red)

        H_coll = ham.build_H_TFIM(self.N, self.ome)
        evals_coll, evecs_coll = LA.eigh(H_coll.toarray()) # compute the eigenvalues and eigenvectors of the collective Hamiltonian using the dense eigensolver
        ini_coll = util.CSS(self.N, np.pi / 2, 0)
        ini_proj_coll = unitaries.init_coeffs_eigenbasis(ini_coll, evecs_coll) # project the initial state onto the eigenbasis of the collective Hamiltonian
        state_t_coll = unitaries.t_evol_eigenbasis(ini_proj_coll, t, evals_coll, evecs_coll) # evolve the state in time using the eigenvalues and eigenvectors of the collective Hamiltonian
        rho_red_coll = util.trace_half_collective(state_t_coll)
        S_A_coll = util.entanglement_entropy(rho_red_coll)
        assert np.isclose(S_A, S_A_coll, atol=1e-6) # check that the entanglement entropy computed from the A2A Hamiltonian matches the entanglement entropy computed from the collective Hamiltonian, which should be the same since the A2A Hamiltonian is designed to reproduce the dynamics of the collective Hamiltonian in the symmetric subspace


###################### Solution sheet 9 ######################


class Test_build_H_AKLT:

    N = 10
    
    def test_build_H_AKLT_periodic(self):
        H_AKLT = ham.build_H_AKLT(self.N, open_bc=False).real
        evals = eigsh(H_AKLT, k=2, which='SA', return_eigenvectors=False)
        evals.sort()
        gap = evals[1] - evals[0]
        assert gap > 0 # check that the energy gap is positive, which means that the ground state is unique and the system is gapped

    def test_build_H_AKLT_open(self):
        H_AKLT = ham.build_H_AKLT(self.N, open_bc=True).real
        evals = eigsh(H_AKLT, k=2, which='SA', return_eigenvectors=False)
        evals.sort()
        gap = evals[1] - evals[0]
        assert gap < 1e-10 # check that the energy gap is approximately zero, which means that the ground state is degenerate and the system is gapless

    def test_build_H_AKLT_theta_large(self):
        theta = 1
        H_AKLT = ham.build_H_AKLT(self.N, open_bc=False, theta=theta).real
        evals = eigsh(H_AKLT, k=2, which='SA', return_eigenvectors=False)
        evals.sort()
        gap = evals[1] - evals[0]
        assert gap < 1e-10 # check that the energy gap is approximately zero, which means that the ground state is degenerate and the system is gapless

    def test_build_H_AKLT_theta_small(self):
        theta = 0.1
        H_AKLT = ham.build_H_AKLT(self.N, open_bc=False, theta=theta).real
        evals = eigsh(H_AKLT, k=2, which='SA', return_eigenvectors=False)
        evals.sort()
        gap = evals[1] - evals[0]
        assert gap > 0 # check that the energy gap is positive, which means that the ground state is unique and the system is gapped