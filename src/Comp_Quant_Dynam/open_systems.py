

import numpy as np
import numpy.linalg as LA

from Comp_Quant_Dynam.utility import pij
from Comp_Quant_Dynam.integrators import Euler_step
from Comp_Quant_Dynam.hamiltonians import build_H_EIT


##################### Exercise sheet 11 ######################


def tr_reduce_L(L_mat):
    """
    Reduces the Liouvillian matrix `L_mat` to account for the trace condition Tr(rho) = 1 when solving for the steady state density matrix.
    The function constructs a reduced Liouvillian matrix `L_mat_red` and a corresponding vector `b_vec` such that the steady state can be found by solving the linear system L_mat_red * rho_ss = b_vec. The reduction is performed by eliminating the first row and column of the Liouvillian matrix and adjusting the last column to account for the trace condition.
    """
    
    dim_L = len(L_mat)
    dim_H = int(np.sqrt(dim_L))
    L_mat_red = np.copy(L_mat[1:, 1:])
    b_vec = np.zeros((dim_L - 1,), dtype='complex')
    for i in range(1, dim_L):
        for k in range(1, dim_H):
            L_mat_red[i - 1, -1 + k * (dim_H + 1)] -= L_mat[i, 0]
        b_vec[i - 1] = -L_mat[i, 0]
    return L_mat_red, b_vec

def rho_ss(L_mat):
    """
    Calculate the steady state density matrix for a given Liouvillian matrix `L_mat`. The steady state is obtained by solving the linear system L * rho_ss = 0, subject to the trace condition Tr(rho_ss) = 1.
    The function first reduces the Liouvillian matrix to account for the trace condition, then solves the resulting linear system to find the steady state vector, which is reshaped into a density matrix form.
    """

    dim_L = len(L_mat)
    dim_H = int(np.sqrt(dim_L))
    L_mat_red, b_vec = tr_reduce_L(L_mat)
    ss = LA.solve(L_mat_red, b_vec)
    ss_full = np.zeros((dim_L,), dtype='complex')
    ss_full[0] = 1
    for k in range(1, dim_H):
        ss_full[0] -= ss[-1 + k * (dim_H + 1)]
    ss_full[1:] = ss
    ss_mat = ss_full.reshape((dim_H, dim_H))
    return ss_mat


###################### Solution sheet 11 ######################


def ME_RHS(rho_vec, H_mat, Lindblad_terms):
    """
    Returns the right-hand side of the master equation for a given density matrix `rho_vec`, Hamiltonian `H_mat`, and list of Lindblad operators `Lindblad_terms`.
    The master equation is given by:
    d(rho)/dt = -i [H, rho]
    + sum_i (L_i rho L_i^dagger - 1/2 {L_i^dagger L_i, rho})
    where [H, rho] is the commutator of H and rho, and {L_i^dagger L_i, rho} is the anti-commutator of L_i^dagger L_i and rho. The density matrix `rho_vec` is represented as a vector in the Liouville space, and the Hamiltonian `H_mat` and Lindblad operators `Lindblad_terms` are represented as matrices in the Hilbert space.
    The output is a vector in the Liouville space representing the time derivative of the density matrix.
    """
    dim = len(H_mat)
    rho = rho_vec.reshape((dim, dim))
    rho_out = -1j * (H_mat @ rho - rho @ H_mat)
    for i in range(len(Lindblad_terms)):
        G = Lindblad_terms[i]
        G_T_conj = G.T.conjugate()
        rho_out += G @ rho @ G_T_conj - 1 / 2 *  (G_T_conj @ G @ rho + rho @ G_T_conj @ G)
    return rho_out.reshape((dim ** 2,))

def build_EIT_operators(params):
    """
    Builds the Hamiltonian and Lindblad operators for a three-level lambda system with given parameters `params`.
    The Hamiltonian and Lindblad operators are constructed from the Rabi frequencies, decay rates, and detunings of the system.
    The parameters `params` should be a dictionary containing the following keys:
    - "omegaP": Rabi frequency of the pump field
    - "omegaC": Rabi frequency of the control field
    - "gammaP": Decay rate of the excited state to the ground state g1
    - "gammaC": Decay rate of the excited state to the ground state g2
    - "Gamma": Decay rate of the ground state g2 to the ground state g1
    - "DeltaP": Detuning of the pump field
    - "DeltaC": Detuning of the control field
    """

    # build the Hamiltonian
    H = build_H_EIT(params)
    dim_H = len(H)

    # build jump operators
    Lp = np.sqrt(params["gammaP"]) * pij(dim_H, 0, 2)
    Lc = np.sqrt(params["gammaC"]) * pij(dim_H, 1, 2)
    Lg = np.sqrt(params["Gamma"]) * pij(dim_H, 0, 1)
    L_list = np.array([Lp, Lc, Lg])
    
    return H, L_list

def build_EIT_operators_ladder_scheme(params):
    """
    Builds the Hamiltonian and Lindblad operators for a three-level ladder-scheme system with given parameters `params`.
    In contrast to the lambda system, we have decay from the g1 state to the exited state and from excited state to the g2 state.
    The Hamiltonian and Lindblad operators are constructed from the Rabi frequencies, decay rates, and detunings of the system.
    The parameters `params` should be a dictionary containing the following keys:
    - "omegaP": Rabi frequency of the pump field
    - "omegaC": Rabi frequency of the control field
    - "gammaP": Decay rate of the g1 state to the excited state
    - "gammaC": Decay rate of the excited state to the g2 states
    - "Gamma": Decay rate of the ground state g2 to the ground state g1 - not used in this case
    - "DeltaP": Detuning of the pump field
    - "DeltaC": Detuning of the control field
    """

    # build the Hamiltonian
    H = build_H_EIT(params)
    dim_H = len(H)

    # build jump operators
    Lp = np.sqrt(params["gammaP"]) * pij(dim_H, 2, 0) # inverted compared to EIT case
    Lc = np.sqrt(params["gammaC"]) * pij(dim_H, 1, 2) # same as in EIT case
    Lg = np.sqrt(params["Gamma"]) * pij(dim_H, 0, 1) # not used
    L_list = np.array([Lp, Lc, Lg])
    
    return H, L_list

def build_L_mat_EIT(params):
    """
    Builds the Liouvillian matrix for a three-level system with given parameters `params`.
    The Liouvillian matrix is constructed from the Hamiltonian and Lindblad operators, and represents the time evolution of the density matrix in the Liouville space.
    The parameters `params` should be a dictionary containing the following keys:
    - "omegaP": Rabi frequency of the pump field
    - "omegaC": Rabi frequency of the control field
    - "gammaP": Decay rate of the excited state to the ground state g1
    - "gammaC": Decay rate of the excited state to the ground state g2
    - "Gamma": Decay rate of the ground state g2 to the ground state g1
    - "DeltaP": Detuning of the pump field
    - "DeltaC": Detuning of the control field
    """

    H, L_list = build_EIT_operators(params)

    # build Liouvillian
    dim_H = H.shape[0] 
    dim_L = dim_H ** 2
    L_mat = np.zeros((dim_L, dim_L), dtype='complex')
    for i in range(dim_L):
        L_mat[:, i] = ME_RHS(pij(dim_H ,i // dim_H, i % dim_H).reshape((dim_L,)), H, L_list)
        
    return L_mat

def build_H_non_herm(H, Lindblad_terms):
    """
    Builds the non-Hermitian Hamiltonian for a system with given Hamiltonian `H` and Lindblad operators `Lindblad_terms`.
    The non-Hermitian Hamiltonian is given by:
    H_non_herm = H - i/2 sum_i L_i^dagger L_i
    where H is the Hermitian Hamiltonian, and L_i are the Lindblad operators.
    """

    H_non_herm = H.copy()
    for i in range(len(Lindblad_terms)):
        G = Lindblad_terms[i]
        G_T_conj = G.T.conjugate()
        H_non_herm -= 1j / 2 * (G_T_conj @ G)
    return H_non_herm

def get_jump_probs(L_list, psi, dt):
    """
    Calculates the jump probabilities for a given state vector `psi`, list of Lindblad operators `L_list`, and time step `dt`.
    The jump probabilities are given by:
    p_i = dt * <psi| L_i^dagger L_i |psi>
    where L_i are the Lindblad operators, and <psi| is the conjugate transpose of the state vector `psi`.
    The output is a vector of jump probabilities for each Lindblad operator in `L_list`.
    """

    jump_probs = np.zeros((len(L_list),))
    for i in range(len(L_list)):
        jump_probs[i] = np.real(dt * psi.conjugate().T @ L_list[i].conjugate().T @ L_list[i] @ psi)
    return jump_probs

def trajectory_step(Hnh, L_list, psi, dt):
    """
    Evolves the state vector `psi` for a single time step `dt` using the non-Hermitian Hamiltonian `Hnh` and the list of Lindblad operators `L_list`.
    The evolution is performed by first calculating the jump probabilities for each Lindblad operator, then randomly selecting whether a jump occurs or not based on these probabilities. If a jump occurs, the corresponding Lindblad operator is applied to the state vector. If no jump occurs, the state vector is evolved using the non-Hermitian Hamiltonian. The state vector is then renormalized to ensure it remains a valid quantum state.
    """

    jump_probs = get_jump_probs(L_list, psi, dt)
    r = np.random.uniform()
    which_jump = 0
    while which_jump < len(jump_probs):
        if r > np.sum(jump_probs[0 : which_jump]) and r < np.sum(jump_probs[0 : which_jump + 1]):
            break
        which_jump += 1
    if which_jump == len(jump_probs):
        # Euler step with Hnh
        psi = Euler_step(psi, Hnh, dt, [])
    else:
        # apply jump operator
        psi = L_list[which_jump] @ psi
    # renormalize
    psi = psi / np.sqrt(psi.conjugate() @ psi)
    return psi

def get_trajectory(Hnh, L_list, ini, tvec):
    """
    Evolves the state vector `ini` over a time vector `tvec` using the non-Hermitian Hamiltonian `Hnh` and the list of Lindblad operators `L_list`.
    Returns a matrix `psivec` where each row corresponds to the state vector at a given time in `tvec`. Each time step models the evolution of the quantum state under the influence of the Hamiltonian and possible quantum jumps due to the Lindblad operators.
    """

    dt = tvec[1] - tvec[0]
    psi_vec = np.zeros((len(tvec), len(Hnh)), dtype='complex')
    # store initial wave function
    psi_vec[0] = ini
    psi_curr = ini
    for i in range(1, len(tvec)):
        psi_curr = trajectory_step(Hnh, L_list, psi_curr, dt)
        psi_vec[i] = psi_curr
    return psi_vec