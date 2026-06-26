import numpy as np   # standard numerics library
import math
from scipy.special import hermite as herm
import scipy.sparse as sparse # routines for sparse matrices

from Comp_Quant_Dynam.utility import state2idx, idx2state
from Comp_Quant_Dynam.operators import diagonal_op_sparse, n_party_op_sparse, x_operator_sparse, Sx_sparse, Sz_sparse, Sx_symm, Sz2_symm
import Comp_Quant_Dynam.operators as ops

###################### Solution sheet 1 ######################

def HO_eigenstates_exact(n, x):
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


###################### Solution sheet 2 ######################

def H_kinetic_sparse(x):
    """
    Returns the kinetic energy operator of the quantum harmonic oscillator in the position basis for a grid 'x' as a sparse matrix.
    The kinetic energy operator is represented as a finite difference matrix approximating the second derivative, which is given by the formula:
    T = - (ħ^2 / 2m) * d^2/dx^2, or in numerical units, T = -0.5 * d^2/dx^2. The second derivative can be approximated using the central difference formula:
    d^2ψ/dx^2 ≈ (ψ(x + dx) - 2ψ(x) + ψ(x - dx)) / (dx^2).
    """

    n_points = len(x) # number of grid points
    dx = x[1] - x[0] # grid spacing

    main_diag = np.ones(n_points)
    off_diag = -0.5 * np.ones(n_points - 1)
    H_kin = sparse.diags_array(
        [main_diag, off_diag, off_diag],
        offsets = (0, 1, -1)
    ) / (dx * dx)
    return H_kin

def HO_potential_sparse(x):
    """
    Returns the potential energy operator of the quantum harmonic oscillator in the position basis for a grid 'x' as a sparse matrix.
    The potential energy operator is represented as a diagonal matrix with elements given by V(x) = 0.5 * x^2.
    """
    
    return sparse.diags_array(0.5 * x ** 2)


###################### Solution sheet 3 ######################

def step_potential(x, V0):
    """
    Returns the potential energy of a step potential of step height 'V0' in the position basis for a grid 'x' as a 1D array.
    The step potential is defined as V(x) = 0 for x < 0 and V(x) = V0 for x >= 0.
    """
    potential = V0 * (x >= 0).astype(float)
    return potential

def barrier_potential(x, V0, width):
    """
    Returns the potential energy of a barrier potential of height 'V0' and width 'width' in the position basis for a grid 'x' as a 1D array.
    The barrier potential is defined as V(x) = 0 for |x| > width/2 and V(x) = V0 for |x| <= width/2.
    """
    potential = V0 * ((x >= -width/2) & (x <= width/2)).astype(float)
    return potential


###################### Exercise sheet 4 ######################

def build_H_coupled_HO_man(N1, N2, lam):
    """
    Manually builds the Hamiltonian matrix for two coupled harmonic oscillators in the number basis, where `N1` and `N2` are the maximum occupation numbers for the two oscillators, and `lam` is the coupling strength.
    The Hamiltonian is given by:
    H = H_1 + H_2 + V = 1 / (2m) * (p_1^2 + p_2^2) + k / 2 * (x_1^2 + x_2^2) + lam / 2 * (x_1 - x_2)^2
    This function is inteded for testing purposes, and better implemented using the ladder operators.
    """
    # build a vector of data-index pairs for the non-zero matrix elements
    values = np.array([])
    row = np.array([])
    col = np.array([])
    for i in range(N1*N2):
        state = idx2state(N1,N2,i)
        # diagonal term (k=m=hbar=1)
        values = np.append(values,(1+lam/2)*(1+state[0]+state[1]))
        row = np.append(row,i)
        col = np.append(col,i)
        # off-diagonal elements of a column (8 terms in the perturbing Hamiltonian)
        # a1^dag a1^dag
        stateCoupleTo = [state[0] + 2, state[1] + 0]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[0]+1)*(state[0]+2)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a1
        stateCoupleTo = [state[0] - 2, state[1] + 0]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[0])*(state[0]-1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a2^dag a2^dag
        stateCoupleTo = [state[0] + 0, state[1] + 2]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[1]+1)*(state[1]+2)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a2 a2
        stateCoupleTo = [state[0] + 0, state[1] - 2]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,(lam/4)*np.sqrt((state[1])*(state[1]-1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1^dag a2^dag
        stateCoupleTo = [state[0] + 1, state[1] + 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0]+1)*(state[1]+1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1^dag a2
        stateCoupleTo = [state[0] + 1, state[1] - 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0]+1)*(state[1])))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a2^dag
        stateCoupleTo = [state[0] - 1, state[1] + 1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0])*(state[1]+1)))
            row = np.append(row,indTo)
            col = np.append(col,i)
        # a1 a2
        stateCoupleTo = [state[0] - 1, state[1] -1]
        indTo = state2idx(N1,N2,stateCoupleTo)
        if indTo >= 0:
            values = np.append(values,-(lam/2)*np.sqrt((state[0])*(state[1])))
            row = np.append(row,indTo)
            col = np.append(col,i)
    H = sparse.coo_matrix((values, (row, col)), shape=(N1*N2, N1*N2))
    H.eliminate_zeros() # remove explicit zeros from the sparse matrix representation, which can arise from the way we build the matrix and can cause issues with the eigensolver
    return H


###################### Solution sheet 4 ######################

def coupled_HO_potential(x, y, lam):
    """
    Returns the potential energy of two coupled harmonic oscillators in the position basis for a grid `x` and `y` as a 2D array, where `lam` is the coupling strength.
    The potential energy is given by:
    V(x, y) = (x^2 + y^2) / 2 + lam / 2 * (x - y)^2
    """
    
    potential = (x ** 2 + y ** 2) / 2 + lam / 2 * (x - y) ** 2
    return potential

def class_traj(lam, ini, t):
    """
    Returns the classical trajectories of two coupled harmonic oscillators for a given coupling strength `lam`, initial conditions `ini`, and time `t`.
    The initial conditions are given in the format `ini = (x10, x20, v10, v20)`, where `x10` and `x20` are the initial positions of the two oscillators,
    and `v10` and `v20` are the initial velocities of the two oscillators.
    """
    
    ome_CM = 1 # frequency of the center of mass motion
    ome_rel = np.sqrt(1+2*lam) # frequency of the relative motion, which depends on the coupling strength
    xcm0 = (ini[0] + ini[1]) / 2 # initial position of the center of mass
    vcm0 = (ini[2] + ini[3]) / 2 # initial velocity of the center of mass
    xrel0 = (ini[0] - ini[1]) # initial relative position
    vrel0 = (ini[2] - ini[3]) # initial relative velocity

    xcm = xcm0 * np.cos(ome_CM * t) + vcm0 / ome_CM * np.sin(ome_CM * t) # trajectory of the center of mass
    xrel = xrel0 * np.cos(ome_rel * t) + vrel0 / ome_rel * np.sin(ome_rel * t) # trajectory of the relative motion
    x1 = xcm + xrel / 2 # trajectory of the first oscillator
    x2 = xcm - xrel / 2 # trajectory of the second oscillator
    return x1, x2

def build_H_coupled_HO_improved(N1, N2, lam):
    """
    Builds the Hamiltonian matrix for two coupled harmonic oscillators in the number basis using the ladder operators, where `N1` and `N2` are the maximum occupation numbers for the two oscillators, and `lam` is the coupling strength.
    The Hamiltonian is given by:
    H = H_1 + H_2 + V = 1 / (2m) * (p_1^2 + p_2^2) + k / 2 * (x_1^2 + x_2^2) + lam / 2 * (x_1 - x_2)^2
    This function is more efficient than the manual construction of the Hamiltonian matrix, as it leverages the structure of the ladder operators and avoids explicit loops over the basis states.
    """

    evals_H1 = np.arange(N1) + 0.5
    evals_H2 = np.arange(N2) + 0.5
    H1 = diagonal_op_sparse(evals_H1)
    H2 = diagonal_op_sparse(evals_H2)

    H1_full = n_party_op_sparse([N1, N2], 0, H1)
    H2_full = n_party_op_sparse([N1, N2], 1, H2)

    x1_local = x_operator_sparse(N1)
    x2_local = x_operator_sparse(N2)

    x1_full = n_party_op_sparse([N1, N2], 0, x1_local)
    x2_full = n_party_op_sparse([N1, N2], 1, x2_local)

    delta_x = x1_full - x2_full
    H_coupling = lam / 2 * (delta_x) @ (delta_x) # coupling term is lam/2 * (x1 - x2)^2

    H = H1_full + H2_full + H_coupling
    return H

def coupled_HO_E0_exact(lam):
    """
    Returns the exact ground state energy of two coupled harmonic oscillators with coupling strength `lam`.
    """

    return 1 / 2 + np.sqrt( 1 + 2 * lam) / 2

def coupled_HO_eigenenergies_exact(n_cm, n_rel, lam):
    """
    Returns the exact eigenenergies of two coupled harmonic oscillators for given quantum numbers `n_cm` and `n_rel`
    corresponding to the center of mass and relative motion, respectively, and coupling strength `lam`.
    """

    E_cm = 1 / 2 + n_cm # energy of the center of mass motion, which is unaffected by the coupling
    E_rel = np.sqrt(1 + 2 * lam) * (1 / 2 + n_rel) # energy of the relative motion, which depends on the coupling strength
    return E_cm + E_rel

def HO_product_eigenstates(N1, N2, xgrid):
    """
    Returns the product eigenstates of two harmonic oscillators on a spatial grid.
    """
    
    dim = N1 * N2
    basis_state_pos = np.zeros((dim, len(xgrid), len(xgrid)), dtype=complex)
    for k in range(dim):
        state_ij = idx2state(N1, N2, k)
        state_1 = HO_eigenstates_exact(state_ij[0], xgrid).reshape(len(xgrid),1)
        state_2 = HO_eigenstates_exact(state_ij[1], xgrid).reshape(1,len(xgrid))
        basis_state_pos[k] = np.kron(state_1, state_2)
    return basis_state_pos


###################### Solution sheet 5 ######################

def build_H_TFIM(N, ome):
    """
    Builds the Hamiltonian matrix for the transverse field Ising model (TFIM) for `N` spin-1/2 particles and transverse field strength `ome` as a sparse matrix.
    The Hamiltonian is given by:
    H = -1/N * Sz^2 - ome * Sx
    where Sx and Sz are the collective spin operators in the x and z directions, respectively.
    """
    
    Sx = Sx_sparse(N)
    Sz = Sz_sparse(N)
    H = -Sz @ Sz / N - ome * Sx
    return H

def build_H_TFIM_symm(N, ome):
    """
    Builds the Hamiltonian matrix for the transverse field Ising model (TFIM) for `N` spin-1/2 particles in the positive symmetric subspace and transverse field strength `ome` as a sparse matrix.
    The Hamiltonian is given by:
    H = -1/N * Sz^2 - ome * Sx
    where Sx and Sz are the collective spin operators in the x and z directions, respectively.
    """
    
    Sx = Sx_symm(N)
    Sz2 = Sz2_symm(N)
    H_symm = -Sz2 / N - ome * Sx
    return H_symm
    

###################### Solution sheet 7 ######################

def E_MF(z, phi, omega):
    """
    Returns the mean-field energy of the transverse field Ising model (TFIM) for a given magnetization `z`, phase `phi`, and transverse field strength `omega`.
    The mean-field energy is given by:
    E_MF(z, phi) = -z^2 / 2 - omega * sqrt(1 - z^2) * cos(phi)
    where z is the magnetization along the z-axis, phi is the phase of the transverse magnetization in the x-y plane, and omega is the strength of the transverse field.
    """
    r2 = 1 - z ** 2
    r2 = np.maximum(r2, 1e-10) # avoid numerical issues when z is close to 1 or -1, which would lead to r being close to zero and causing instability in the calculation of the mean-field energy
    return -z ** 2 / 2 - omega * np.sqrt(r2) * np.cos(phi)


###################### Solution sheet 8 ######################

def build_H_TFIM_individual(N, ome):
    """
    Builds the Hamiltonian matrix for the transverse field Ising model (TFIM) for `N` spin-1/2 particles and transverse field strength `ome` as a sparse matrix
    The Hamiltonian is given by:
    H = - sum_i sigma_z^i sigma_z^{i+1} - ome * sum_i sigma_x^i
    where sigma_z^i and sigma_x^i are the Pauli z and x operators acting on the i-th particle, respectively, and we assume periodic boundary conditions, i.e., sigma_z^N = sigma_z^0.
    """
    
    dims_local = [2] * N
    sig_x = [n_party_op_sparse(dims_local, i, ops.sigma_x_sparse()) for i in range(N)]
    sig_z = [n_party_op_sparse(dims_local, i, ops.sigma_z_sparse()) for i in range(N)]

    dim_global = 2 ** N
    H = sparse.csr_array((dim_global, dim_global), dtype=complex)
    for i in range(N):
        H -= ome * sig_x[i]
        H -= sig_z[i] @ sig_z[(i + 1) % N] # periodic boundary conditions
    return H

def E_TFIM_individual_exact(N, B):
    """
    Returns the exact ground state energy of the transverse field Ising model (TFIM) for `N` spin-1/2 particles and transverse field strength `B`.
    The exact ground state energy can be calculated using the Jordan-Wigner transformation, which maps the spin-1/2 system to a system of free fermions, and then diagonalizing the resulting quadratic fermionic Hamiltonian. The exact ground state energy is given by:
    E_exact = - sum_k sqrt(1 + B^2 + 2 * B * cos(2 * pi * k / N))
    where the sum is taken over the momentum modes k = -(N-1)/2, ..., (N-1)/2.
    """
    assert N % 2 == 0, "The exact solution for the ground state energy of the TFIM is only implemented for even N"
    cosine_term = 2 * B * np.cos(2 * np.pi / N * np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1))
    E_exact = -np.sum(np.sqrt(1 + B ** 2 + cosine_term)) # analytical ground state energy
    return E_exact

def build_H_TFIM_A2A(N, B):
    """
    Builds the Hamiltonian matrix for the all-to-all transverse field Ising model (TFIM) for `N` spin-1/2 particles and transverse field strength `B` as a sparse matrix.
    The Hamiltonian is given by:
    H = -1/N * Sz^2 - B * Sx
    where Sx and Sz are the collective spin operators in the x and z directions, respectively.
    """

    dim_global = 2 ** N
    sxis, syis, szis = ops.build_single_spin_ops_sparse(N)
    H_mat = sparse.csr_matrix((dim_global, dim_global))
    for i in range(N):
        for j in range(N):
            H_mat -= szis[i] @ szis[j] / N # interaction term
        H_mat -= B * sxis[i] # field term
    #H_mat = H_mat + H_mat.T - sparse.diags(H_mat.diagonal()) # make it Hermitian
    return H_mat


###################### Solution sheet 9 ######################


def build_H_AKLT(N, theta=np.arctan(1 / 3), open_bc=False):
    """
    Builds the Hamiltonian matrix for the AKLT model for `N` spin-1 particles and angle `theta` as a sparse matrix.
    The Hamiltonian is given by:
    H = sum_i (cos(theta) / (3 * cos(theta0)) + cos(theta) / (2 * cos(theta0)) * S_i * S_{i+1} + sin(theta) / (2 * cos(theta0)) * (S_i * S_{i+1})^2)
    where S_i is the spin-1 operator acting on the i-th particle, and we assume periodic boundary conditions if `open_bc` is False, and open boundary conditions if `open_bc` is True.
    """
    theta0 = np.arctan(1  / 3)
    local_dim = 3
    local_dims = [local_dim] * N
    H_mat = sparse.csr_array((local_dim ** N, local_dim ** N))

    spSp, smSp, szSp = ops.build_single_spin_1_ops_sparse()

    c0 = np.cos(theta) / 3 / np.cos(theta0)
    c1 = np.cos(theta) / 2 / np.cos(theta0)
    c2 = np.sin(theta) / 2 / np.cos(theta0)

    I =  sparse.identity(local_dim ** N)

    end_idx = N - 1 if open_bc else N
    for idx in range(end_idx):
        # first build S_i*S_(i+1)
       
        next_idx = np.mod(idx + 1, N)
        idxs = [idx, next_idx]
        SiSi_p1 = ops.n_party_op_sparse(local_dims, idxs, [szSp, szSp])
        SiSi_p1 += 1/2 * ops.n_party_op_sparse(local_dims, idxs, [spSp, smSp])
        SiSi_p1 += 1/2 * ops.n_party_op_sparse(local_dims, idxs, [smSp, spSp])
        H_mat += c1 * SiSi_p1 
        H_mat += c2 * (SiSi_p1 @ SiSi_p1)
    H_mat += c0 * I * N
    
    return H_mat