import numpy as np
from scipy import sparse

import Comp_Quant_Dynam.utility as util

###################### Solution sheet 4 ######################

def diagonal_op_sparse(arr, offsets = 0):
    """
    Returns a sparse diagonal matrix from a given 1D array `arr`.
    The diagonal matrix has the elements of `arr` on its diagonal and zeros elsewhere.
    """
    
    return sparse.diags_array(arr, offsets=offsets)


def n_party_op_sparse(local_dims, idxs, ops):
    """
    Returns a sparse operator for an n-party system, where `local_dims` is a list of the local dimensions of each party. The `idxs` and `ops` are the indices and the local operators for the parties on which the operator acts non-trivially, respectively.
    However, the function can also take a single operator and a single index, in which case it will return the full operator for that single local operator.
    """
    
    n_ops, ops = util._check_if_sized(ops)

    if n_ops == 1:
        ops = [ops]
        idxs = np.array([idxs])
    assert len(idxs) == n_ops, "Number of indices must match the number of operators."
    assert np.all([idx < len(local_dims) for idx in idxs]), "Index of the local operator must be less than the total number of parties."
    assert np.all([op.shape == (local_dims[idx], local_dims[idx]) for idx, op in zip(idxs, ops)]), "Local operator must have the correct shape corresponding to the local dimension of the party."
    idxs = np.array(idxs)
    asort = np.argsort(idxs)
    idxs = idxs[asort]
    ops = [ops[i] for i in asort]
    eye_left = sparse.eye_array(np.prod(local_dims[:idxs[0]])) # identity operator on the left of the first local operator
    eye_right = sparse.eye_array(np.prod(local_dims[idxs[-1] + 1:])) # identity operator on the right of the last local operator
    full_op = eye_left
    for counter, (idx, op) in enumerate(zip(idxs, ops)):
        full_op = sparse.kron(full_op, op) # Kronecker product of the current full operator and the local operator
        if counter < len(ops) - 1: # if this is not the last
            next_idx = idxs[counter + 1]
            eye_next = sparse.eye_array(np.prod(local_dims[idx + 1:next_idx])) # identity operator between the current local operator and the next local operator
            full_op = sparse.kron(full_op, eye_next) # Kronecker product of the current full operator and the identity operator between the current local operator and the next local operator
    full_op = sparse.kron(full_op, eye_right)

    #eye_left = sparse.eye_array(np.prod(local_dims[:idx])) # identity operator on the left of the local operator
    #eye_right = sparse.eye_array(np.prod(local_dims[idx+1:])) # identity operator on the right of the local operator
    #full_op = sparse.kron(eye_left, sparse.kron(op, eye_right)) # full operator is the Kronecker product of the left identity, local operator, and right identity
    return full_op

def a_operator_sparse(N):
    """
    Returns the annihilation ladder operator for a system with a local dimension `N` as a sparse matrix.
    The annihilation operator is defined as:
    a |n> = sqrt(n) |n-1> for n > 0
    a |0> = 0
    """
    vals = np.sqrt(np.arange(1, N)) # non-zero elements of the annihilation operator
    a_op = diagonal_op_sparse(vals, offsets=1) # create a sparse diagonal matrix with the non-zero elements on the first upper diagonal
    return a_op

def adag_operator_sparse(N):
    """
    Returns the creation ladder operator for a system with a local dimension `N` as a sparse matrix.
    The creation operator is defined as:
    a^dag |n> = sqrt(n+1) |n+1> for n < N-1
    a^dag |N-1> = 0
    """
    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = a_op.transpose().conjugate() # the creation operator is the Hermitian conjugate of the annihilation operator
    return adag_op

def n_operator_sparse(N):
    """
    Returns the number operator for a system with a local dimension `N` as a sparse matrix.
    The number operator is defined as:
    n |n> = n |n>
    """
    n_vals = np.arange(N, dtype=float) # eigenvalues of the number operator
    n_op = diagonal_op_sparse(n_vals) # create a sparse diagonal matrix with the eigenvalues on the diagonal
    return n_op

def x_operator_sparse(N):
    """
    Returns the position operator for a system with a local dimension `N` as a sparse matrix.
    The position operator is defined as:
    x |n> = sqrt(1/2) (a + a^dag) |n>
    """

    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = adag_operator_sparse(N) # get the creation operator
    x_op = np.sqrt(1/2) * (a_op + adag_op) # the position operator is the sum of the annihilation and creation operators
    return x_op

def p_operator_sparse(N):
    """
    Returns the momentum operator for a system with a local dimension `N` as a sparse matrix.
    The momentum operator is defined as:
    p |n> = i * sqrt(1/2) (a^dag - a) |n>
    """

    a_op = a_operator_sparse(N) # get the annihilation operator
    adag_op = adag_operator_sparse(N) # get the creation operator
    p_op = 1j * np.sqrt(1/2) * (adag_op - a_op) # the momentum operator is the difference of the creation and annihilation operators, multiplied by i and sqrt(1/2)
    return p_op

def n_proj_operator_sparse(local_dims, idx, n):
    """
    Returns the projector operator for the `n`-th state of the `idx`-th party in an system with local dimensions given by `local_dims` as a sparse matrix.
    The projector operator is defined as:
    P_n = |n><n| for the n-th state of the idx-th party, and acts as the identity on the other parties.
    """

    proj_arr = np.eye(1, local_dims[idx], n)
    proj_op = diagonal_op_sparse(proj_arr.flatten()) # create a sparse diagonal matrix with the non-zero element on the diagonal corresponding to the n-th state
    projector_full = n_party_op_sparse(local_dims, idx, proj_op) # create the full projector operator for the n-th state of the idx-th party
    return projector_full


###################### Solution sheet 5 ######################

def Sx_sparse(N):
    """
    Returns the Sx operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sx operator is defined as:
    Sx = (S+ + S-) / 2
    where S+ and S- are the raising and lowering operators, respectively.
    """
    
    S_plus_vec = np.sqrt((N - np.arange(0, N)) * (np.arange(0, N) + 1)) 
    S_plus = sparse.diags_array(S_plus_vec, offsets=-1)
    return (S_plus + S_plus.T) / 2

def Sy_sparse(N):
    """
    Returns the Sy operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sy operator is defined as:
    Sy = (S+ - S-) / (2i)
    where S+ and S- are the raising and lowering operators, respectively.
    """
    
    S_plus_vec = np.sqrt((N - np.arange(0, N)) * (np.arange(0, N) + 1)) 
    S_plus = sparse.diags_array(S_plus_vec, offsets=-1)
    return (S_plus - S_plus.T) / (2j)

def Sz_sparse(N):
    """
    Returns the Sz operator for a system of `N` spin-1/2 particles as a sparse matrix.
    The Sz operator is defined as:
    Sz |m> = m |m>
    where m is the magnetic quantum number corresponding to the state |m>.
    """
    
    Sz_vec = np.arange(N + 1) - N / 2 # from -N/2 to N/2 in steps of 1 
    return sparse.diags_array(Sz_vec)

def build_spin_ops_sparse(N):
    """
    Returns the collective spin operators Sx, Sy, and Sz for a system of `N` spin-1/2 particles as sparse matrices.
    The collective spin operators are defined as:
    Sx = sum_i sigma_x^i / 2
    Sy = sum_i sigma_y^i / 2
    Sz = sum_i sigma_z^i / 2
    where sigma_x^i, sigma_y^i, and sigma_z^i are the single-site Pauli operators acting on the i-th particle.
    """

    Sx = Sx_sparse(N)
    Sy = Sy_sparse(N)
    Sz = Sz_sparse(N)
    return Sx, Sy, Sz

def _T_positive_parity_symm(N):
    """
    Returns the basis-change matrix from the full Dicke basis to the
    positive-parity symmetric subspace basis.
    """

    S = N / 2
    full_dim = N + 1
    L = int(np.floor(S)) + 1
    T = np.zeros((full_dim, L), dtype=float)
    for k in range(L):
        m = S - k
        i_pos = int(m + S)
        i_neg = int(-m + S)
        if abs(m) < 1e-12:
            T[i_pos, k] = 1.0
        else:
            T[i_pos, k] = 1.0 / np.sqrt(2.0)
            T[i_neg, k] = 1.0 / np.sqrt(2.0)
    return T

def Sx_symm(N):
    """
    Returns the Sx operator for a system of `N` spin-1/2 particles in the positive symmetric subspace as a sparse matrix.
    The Sx operator is defined as:
    Sx = (S+ + S-) / 2
    where S+ and S- are the raising and lowering operators, respectively.
    """

    T = _T_positive_parity_symm(N)

    Sx_full = Sx_sparse(N)
    Sx_full_arr = Sx_full.toarray() if hasattr(Sx_full, "toarray") else np.array(Sx_full)
    Sx_reduced = T.T @ (Sx_full_arr @ T)
    return sparse.csr_array(Sx_reduced)

def Sz2_symm(N):
    """
    Returns the Sz^2 operator for a system of `N` spin-1/2 particles in the positive symmetric subspace as a sparse matrix.
    The Sz^2 operator is defined as:
    Sz^2 |m> = m^2 |m>
    where m is the magnetic quantum number corresponding to the state |m>.
    Note that Sz vanishes in the positive symmetric subspace.
    """

    T = _T_positive_parity_symm(N)

    Sz_full = Sz_sparse(N)
    Sz_full_arr = Sz_full.toarray() if hasattr(Sz_full, "toarray") else np.array(Sz_full)
    Sz2_reduced = T.T @ ((Sz_full_arr @ Sz_full_arr) @ T)
    return sparse.csr_array(Sz2_reduced)

def sigma_x_sparse():
    """
    Returns the single-site sigma_x operator for a spin-1/2 particle as a sparse matrix.
    The sigma_x operator is defined as:
    sigma_x |0> = |1>
    sigma_x |1> = |0>
    """

    return sparse.csr_array([[0, 1], [1, 0]])

def sigma_y_sparse():
    """
    Returns the single-site sigma_y operator for a spin-1/2 particle as a sparse matrix.
    The sigma_y operator is defined as:
    sigma_y |0> = -i |1>
    sigma_y |1> = i |0>
    """

    return sparse.csr_array([[0, 1j], [-1j, 0]])

def sigma_z_sparse():
    """
    Returns the single-site sigma_z operator for a spin-1/2 particle as a sparse matrix.
    The sigma_z operator is defined as:
    sigma_z |0> = -|0>
    sigma_z |1> = |1>
    """

    return sparse.csr_array([[-1, 0], [0, 1]])


###################### Solution sheet 8 ######################

def build_single_spin_ops_sparse(N):
    """
    Returns the single-site spin operators Sx, Sy, and Sz for a system of `N` spin-1/2 particles as sparse matrices.
    The single-site spin operators are defined as:
    sx^i = sigma_x^i / 2
    sy^i = sigma_y^i / 2
    sz^i = sigma_z^i / 2
    where sigma_x^i, sigma_y^i, and sigma_z^i are the single-site Pauli operators acting on the i-th particle.
    The function returns three lists sxi, syi, and szi, where sxi[i], syi[i], and szi[i] are sx^i, sy^i, and sz^i, respectively, represented as sparse matrices acting on the full Hilbert space of the system.
    """

    sxSp = sigma_x_sparse() / 2
    sySp = sigma_y_sparse() / 2
    szSp = sigma_z_sparse() / 2
    
    dims = [2] * N # local dimensions for each spin-1/2 particle

    sxi=[]
    syi=[]
    szi=[]
    for i in range(N):
        sxi.append(n_party_op_sparse(dims, i, sxSp))
        syi.append(n_party_op_sparse(dims, i, sySp))
        szi.append(n_party_op_sparse(dims, i, szSp))

    return sxi, syi, szi


###################### Solution sheet 9 ######################


def build_single_spin_1_ops_sparse():
    """
    Returns the single-site spin-1 operators S+, S-, and Sz as sparse matrices.
    """

    sm = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) * np.sqrt(2)
    sp = sm.T
    sz = np.array([[-1., 0, 0], [0, 0, 0], [0, 0, 1.]])

    sp_sparse = sparse.csr_array(sp)
    sm_sparse = sparse.csr_array(sm)
    sz_sparse = sparse.csr_array(sz)

    return sp_sparse, sm_sparse, sz_sparse

def get_coeff_MPS(state, a_tensor_arr):
    """
    Returns the coefficient of the product `state` |state> in the MPS representation defined by the list of tensors `a_tensor_arr`.
    The function expects a `state` vector of the form [s1, s2, ..., sN] with si in {-1, 0, 1}, where si corresponds to the local state of the i-th spin-1 particle.
    The coefficient is calculated by contracting the tensors in the MPS representation.
    """

    curr_mat = a_tensor_arr[1 - state[0]]
    for i in range(1, len(state)):
        curr_mat = curr_mat @ a_tensor_arr[1 - state[i]]
    coeff = curr_mat.trace()
    return coeff

def build_E_mat_MPS(a_tensor_arr, op=None):
    """
    Builds the E-matrix for a given list of tensors `A_tensor_list` and a local operator `op`, where each tensor corresponds to a local operator in the MPS representation.
    By default, the identity operator is used, which gives the E-matrix needed for calculating the norm.
    """
    if op is None:
        op = sparse.csr_array(sparse.eye_array(len(a_tensor_arr)))
    assert op.shape == (len(a_tensor_arr), len(a_tensor_arr)), "Operator shape must match the number of tensors"
    shape = sparse.kron(a_tensor_arr[0], a_tensor_arr[0].conjugate()).shape
    E_op_mat = np.zeros(shape, dtype='complex')
    for i in range(len(a_tensor_arr)):
        for j in range(len(a_tensor_arr)):
            if np.isclose(np.abs(op[i, j]), 0):
                continue
            E_op_mat += op[i, j] * sparse.kron(a_tensor_arr[i], a_tensor_arr[j].conjugate())
    return E_op_mat

def corr_func_MPS(N, E_mat, idxs, E_op_arr):
    """
    Computes the expectation value of a the correlation function for a given list of local E-operators `E_op_arr` at the specified indices `idxs` in a system of size `N`, using the E-matrix `E_mat` for all other sites.
    """
    idxs = np.asarray(idxs)
    E_op_arr = np.asarray(E_op_arr)
    asort = np.argsort(idxs)
    idxs = idxs[asort]
    E_op_arr = E_op_arr[asort]

    curr_mat = np.linalg.matrix_power(E_mat, idxs[0])
    for i in range(len(idxs)):
        curr_mat = curr_mat @ E_op_arr[i]
        if i < len(idxs) - 1:
            curr_mat = curr_mat @ np.linalg.matrix_power(E_mat, idxs[i + 1] - idxs[i] - 1)
    curr_mat = curr_mat @ np.linalg.matrix_power(E_mat, N - 1 - idxs[-1])

    return curr_mat.trace()
