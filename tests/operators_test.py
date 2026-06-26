import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

import Comp_Quant_Dynam.hamiltonians as ham
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops


###################### Solution sheet 4 #####################

class Test_ladder_operators:

    N = 20

    a_op = ops.a_operator_sparse(N)
    adag_op = ops.adag_operator_sparse(N)
    n_op = ops.n_operator_sparse(N)
    x_op = ops.x_operator_sparse(N)
    p_op = ops.p_operator_sparse(N)

    def test_a_adag_commutation(self):
        commutator = self.a_op @ self.adag_op - self.adag_op @ self.a_op
        identity = sparse.eye(self.N)
        diff = commutator - identity
        assert np.allclose(diff.data[:-1], 0)

    def test_n_identity(self):
        n_from_a_adag = self.adag_op @ self.a_op
        diff = n_from_a_adag - self.n_op
        assert np.allclose(diff.data, 0)

    def test_x_p_commutation(self):
        commutator = self.x_op @ self.p_op - self.p_op @ self.x_op
        identity = 1j * sparse.eye(self.N)
        diff = commutator - identity
        diff_dense = diff.toarray()
        assert np.allclose(diff_dense[:-1, :-1], 0) # the last row and column deviate due to the truncation of the operators.

    def test_a_adag_hermiticity(self):
        assert np.allclose(self.a_op.data, self.adag_op.conj().T.data)
    
    def test_n_hermiticity(self):
        assert np.allclose(self.n_op.data, self.n_op.data.conj())
    
    def test_x_hermiticity(self):
        diff = self.x_op - self.x_op.T.conj()
        assert np.allclose(diff.data, 0)
    
    def test_p_hermiticity(self):
        diff = self.p_op - self.p_op.T.conj()
        assert np.allclose(diff.data, 0)

class Test_n_proj_operator_sparse:

    N1 = 5
    N2 = 5
    N_vec = [N1, N2]

    n_proj_a_2 = ops.n_proj_operator_sparse(N_vec, 0, 2).toarray()
    n_proj_b_3 = ops.n_proj_operator_sparse(N_vec, 1, 3).toarray()

    state_2_3 = np.kron(np.eye(1, N1, 2).flatten(), np.eye(1, N2, 3).flatten()) # state |2,3>
    state_1_1 = np.kron(np.eye(1, N1, 1).flatten(), np.eye(1, N2, 1).flatten()) # state |1,1>
    state_2_0 = np.kron(np.eye(1, N1, 2).flatten(), np.eye(1, N2, 0).flatten()) # state |2,0>

    def test_n_proj_operator_sparse_0_2(self):
        state_0_2_proj_a = self.n_proj_a_2 @ self.state_2_3
        state_0_2_proj_b = self.n_proj_b_3 @ self.state_2_3
        assert np.allclose(state_0_2_proj_a, self.state_2_3)
        assert np.allclose(state_0_2_proj_b, self.state_2_3)

    def test_n_proj_operator_sparse_1_3(self):
        state_1_3_proj_a = self.n_proj_a_2 @ self.state_1_1
        state_1_3_proj_b = self.n_proj_b_3 @ self.state_1_1
        assert np.allclose(state_1_3_proj_a, 0)
        assert np.allclose(state_1_3_proj_b, 0)

    def test_n_proj_operator_sparse_2_0(self):
        state_2_0_proj_a = self.n_proj_a_2 @ self.state_2_0
        state_2_0_proj_b = self.n_proj_b_3 @ self.state_2_0
        assert np.allclose(state_2_0_proj_a, self.state_2_0)
        assert np.allclose(state_2_0_proj_b, 0)


class Test_n_party_op_sparse:

    local_dims = [4, 5, 6, 5, 4, 3]

    x1 = ops.x_operator_sparse(local_dims[0])
    p2 = ops.p_operator_sparse(local_dims[2])
    n3 = ops.n_operator_sparse(local_dims[4])
    
    x1_p2_n3 = ops.n_party_op_sparse(local_dims, [0, 2, 4], [x1, p2, n3])


    def test_n_party_op_sparse(self):
        # construct the expected operator by hand
        expected_x1_p2_n3 = ops.n_party_op_sparse(self.local_dims, 0, self.x1) @ ops.n_party_op_sparse(self.local_dims, 2, self.p2) @ ops.n_party_op_sparse(self.local_dims, 4, self.n3)
        diff = self.x1_p2_n3 - expected_x1_p2_n3
        assert np.allclose(diff.data, 0)

    def test_n_party_op_sparse_man(self):
        # construct the expected operator by hand
        expected_x1_p2_n3 = sparse.kron(sparse.kron(sparse.kron(sparse.kron(sparse.kron(self.x1, sparse.eye_array(self.local_dims[1])), self.p2), sparse.eye_array(self.local_dims[3])), self.n3), sparse.eye_array(self.local_dims[5]))
        diff = self.x1_p2_n3 - expected_x1_p2_n3
        assert np.allclose(diff.data, 0)

###################### Solution sheet 5 ######################

class Test_Sx_Sy_Sz_sparse:
    
    N = 5

    Sx_op = ops.Sx_sparse(N)
    Sy_op = ops.Sy_sparse(N)
    Sz_op = ops.Sz_sparse(N)

    def test_Sx_hermiticity(self):
        diff = self.Sx_op - self.Sx_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_Sy_hermiticity(self):
        diff = self.Sy_op - self.Sy_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_Sz_hermiticity(self):
        diff = self.Sz_op - self.Sz_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_Sx_trace(self):
        assert np.isclose(self.Sx_op.trace(), 0)

    def test_Sz_trace(self):
        assert np.isclose(self.Sz_op.trace(), 0)

    def test_Sy_trace(self):
        assert np.isclose(self.Sy_op.trace(), 0)

    def test_Sz_Sx_commutation(self):
        commutator = self.Sz_op @ self.Sx_op - self.Sx_op @ self.Sz_op
        diff = commutator - 1j * self.Sy_op
        assert np.allclose(diff.data, 0)

    def test_Sx_Sy_commutation(self):
        commutator = self.Sx_op @ self.Sy_op - self.Sy_op @ self.Sx_op
        diff = commutator - 1j * self.Sz_op
        assert np.allclose(diff.data, 0)

    def test_Sy_Sz_commutation(self):
        commutator = self.Sy_op @ self.Sz_op - self.Sz_op @ self.Sy_op
        diff = commutator - 1j * self.Sx_op
        assert np.allclose(diff.data, 0)
        
class Test_sigma_x_y_z_sparse:

    N = 5

    sigma_x_op = ops.sigma_x_sparse()
    sigma_y_op = ops.sigma_y_sparse()
    sigma_z_op = ops.sigma_z_sparse()

    def test_sigma_x_hermiticity(self):
        diff = self.sigma_x_op - self.sigma_x_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_sigma_y_hermiticity(self):
        diff = self.sigma_y_op - self.sigma_y_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_sigma_z_hermiticity(self):
        diff = self.sigma_z_op - self.sigma_z_op.T.conj()
        assert np.allclose(diff.data, 0)

    def test_sigma_x_trace(self):
        assert np.isclose(self.sigma_x_op.trace(), 0)
    
    def test_sigma_y_trace(self):
        assert np.isclose(self.sigma_y_op.trace(), 0)
    
    def test_sigma_z_trace(self):
        assert np.isclose(self.sigma_z_op.trace(), 0)

    def test_sigma_z_x_commutation(self):
        commutator = self.sigma_z_op @ self.sigma_x_op - self.sigma_x_op @ self.sigma_z_op
        diff = commutator - 2j * self.sigma_y_op
        assert np.allclose(diff.data, 0)

    def test_sigma_y_z_commutation(self):
        commutator = self.sigma_y_op @ self.sigma_z_op - self.sigma_z_op @ self.sigma_y_op
        diff = commutator - 2j * self.sigma_x_op
        assert np.allclose(diff.data, 0)

    def test_sigma_x_y_commutation(self):
        commutator = self.sigma_x_op @ self.sigma_y_op - self.sigma_y_op @ self.sigma_x_op
        diff = commutator - 2j * self.sigma_z_op
        assert np.allclose(diff.data, 0)

    
###################### Solution sheet 8 ######################


class Test_build_single_spin_ops_sparse:

    N = 5
    local_dims = [2] * N
    sxi, syi, szi = ops.build_single_spin_ops_sparse(N)


    def test_sxi(self):
        i = 0
        sx_i = self.sxi[i]
        sx = ops.sigma_x_sparse() / 2
        # construct the expected operator by hand
        expected_sx_i = ops.n_party_op_sparse(self.local_dims, i, sx)
        diff = sx_i - expected_sx_i
        assert np.allclose(diff.data, 0)

    def test_syi(self):
        i = 2
        sy_i = self.syi[i]
        sy = ops.sigma_y_sparse() / 2
        # construct the expected operator by hand
        expected_sy_i = ops.n_party_op_sparse(self.local_dims, i, sy)
        diff = sy_i - expected_sy_i
        assert np.allclose(diff.data, 0)

    def test_szi(self):
        i = 4
        sz_i = self.szi[i]
        sz = ops.sigma_z_sparse() / 2
        # construct the expected operator by hand
        expected_sz_i = ops.n_party_op_sparse(self.local_dims, i, sz)
        diff = sz_i - expected_sz_i
        assert np.allclose(diff.data, 0)


###################### Solution sheet 9 ######################


class Test_build_single_spin_1_ops_sparse:

    sp_sprs, sm_sprs, sz_sprs = ops.build_single_spin_1_ops_sparse()

    sx_sprs = (sp_sprs + sm_sprs) / 2
    sy_sprs = (sp_sprs - sm_sprs) / (2j)
    print("sx = ", sx_sprs.toarray())
    print("sy = ", sy_sprs.toarray())

    def test_trace(self):
        assert np.isclose(self.sp_sprs.trace(), 0)
        assert np.isclose(self.sm_sprs.trace(), 0)
        assert np.isclose(self.sz_sprs.trace(), 0)

    def test_hermiticity(self):
        assert np.allclose(self.sp_sprs.data, self.sm_sprs.conj().T.data)
        assert np.allclose(self.sz_sprs.data, self.sz_sprs.conj().T.data)

    def test_commutation_xy(self):
        commutator = self.sx_sprs @ self.sy_sprs - self.sy_sprs @ self.sx_sprs
        diff = commutator - 1j * self.sz_sprs
        assert np.allclose(diff.data, 0)

    def test_commutation_yz(self):
        commutator = self.sy_sprs @ self.sz_sprs - self.sz_sprs @ self.sy_sprs
        diff = commutator - 1j * self.sx_sprs
        assert np.allclose(diff.data, 0)

    def test_commutation_zx(self):
        commutator = self.sz_sprs @ self.sx_sprs - self.sx_sprs @ self.sz_sprs
        diff = commutator - 1j * self.sy_sprs
        assert np.allclose(diff.data, 0)

class Test_build_E_mat_MPS:

    def test_build_E_mat_MPS(self):
        # Test the build_E_mat_MPS function with a simple example
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0, 1], [1, 0]])
        E = ops.build_E_mat_MPS([A, B])
        expected_E = np.eye(4) + np.fliplr(np.eye(4))
        print("E = ", E)
        print("expected_E = ", expected_E)
        assert np.allclose(E, expected_E)

    def test_build_E_mat_MPS_convergence(self):

        sig_x = ops.sigma_x_sparse()
        sig_y = ops.sigma_y_sparse()
        sig_z = ops.sigma_z_sparse()

        sig_p = (sig_x + 1j * sig_y).real / 2
        sig_m = sig_p.T

        a_tensor_arr = np.array([-np.sqrt(2 / 3) * sig_m, -np.sqrt(1 / 3) * sig_z, np.sqrt(2 / 3) * sig_p])

        E = ops.build_E_mat_MPS(a_tensor_arr)
        # The largest eigenvalue of E should be 1 for a properly normalized MPS
        largest_eigenvalue = np.max(np.abs(np.linalg.eigvals(E)))
        assert np.isclose(largest_eigenvalue, 1.0)

        a_tensor_power = np.linalg.matrix_power(E, 1000)

        norm = np.trace(a_tensor_power)
        assert np.isclose(norm, 1.0)

class Test_build_correlation_function:

   def test_build_correlation_function_hidden_order(self):
       
        sig_x = ops.sigma_x_sparse() 
        sig_y = ops.sigma_y_sparse()
        sig_z = ops.sigma_z_sparse()

        p_sprs, sm_sprs, sz_sprs = ops.build_single_spin_1_ops_sparse()


        sig_p = (sig_x + 1j * sig_y).real / 2
        sig_m = sig_p.T
        


        a_tensor_arr = np.array([-np.sqrt(2 / 3) * sig_m, -np.sqrt(1 / 3) * sig_z, np.sqrt(2 / 3) * sig_p])

        
        E_mat = ops.build_E_mat_MPS(a_tensor_arr).real
        Ez_mat = ops.build_E_mat_MPS(a_tensor_arr, sz_sprs).real
        exp_sz = np.diag(np.exp(1j * np.pi * sz_sprs.diagonal()))
        Eexpz_mat = ops.build_E_mat_MPS(a_tensor_arr, exp_sz).real

        N = 30
    
        string_corr_list = np.zeros((N - 1,))
        for i in range(N - 1):
            idx_arr = np.arange(i + 2)
            op_arr = np.array([Ez_mat if (idx == 0 or idx == i + 1) else Eexpz_mat for idx in idx_arr])
            string_corr_list[i] =  ops.corr_func_MPS(N, E_mat, idx_arr, op_arr)
        assert np.abs(string_corr_list[0]) > 0
        string_corr_list_diff = string_corr_list - string_corr_list[0]
        assert np.allclose(string_corr_list_diff, 0, atol=1e-12)

        
class Test_get_coeff_MPS:
    sig_x = ops.sigma_x_sparse()
    sig_y = ops.sigma_y_sparse()
    sig_z = ops.sigma_z_sparse()

    sig_p = (sig_x + 1j * sig_y).real / 2
    sig_m = sig_p.T

    a_tensor_arr = np.array([-np.sqrt(2 / 3) * sig_m, -np.sqrt(1 / 3) * sig_z, np.sqrt(2 / 3) * sig_p])
    
    def test_get_coeff_MPS_AKLT(self):
        N = 9
        local_dim = 3
        global_dim = local_dim ** N
        H_mat = ham.build_H_AKLT(N)
        evals, evecs = eigsh(H_mat, k=1, which='SA')

        coeffs_mps = np.zeros((global_dim,), dtype=np.complex128)
        for idx in range(global_dim):
            state = util.n_party_idx2state(idx, local_dim, N)
            coeffs_mps[idx] = ops.get_coeff_MPS(state, self.a_tensor_arr)

        coeffs_mps /= np.linalg.norm(coeffs_mps)
        
        overlap = np.abs(np.vdot(evecs[:, 0], coeffs_mps)) ** 2

        assert np.isclose(overlap, 1.0, atol=1e-6)