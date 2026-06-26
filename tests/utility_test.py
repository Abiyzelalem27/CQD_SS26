import numpy as np
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.operators as ops


class Test_example_function:

    def test_example_func_zero(self):
        x = 0
        expected =  1 / np.pi ** (1 / 4)
        result = util.example_func(x)
        assert np.allclose(expected, result)

    def test_example_func_symmetry(self):
        x = np.array([-1, 1])
        result = util.example_func(x)
        assert np.allclose(result[0], result[1])


###################### Solution sheet 2 ######################


class Test_create_xvals:

    L = 10
    npoints = 101
    
    def test_create_xvals_length(self):
        
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert len(xvals) == self.npoints

    def test_create_xvals_range(self):

        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[0], -self.L/2)
        assert np.isclose(xvals[-1], self.L/2)

    def test_create_xvals_spacing(self):
        xvals, dx = util.create_xvals(self.L, self.npoints)
        expected_dx = self.L / (self.npoints - 1)
        assert np.isclose(dx, expected_dx)

    def test_create_xvals_zero_centered(self):
        # only works if npoints is odd
        xvals, dx = util.create_xvals(self.L, self.npoints)
        assert np.isclose(xvals[self.npoints // 2], 0) # check that the middle point is approximately zero


###################### Solution sheet 3 ######################


class Test_FT_iFT:

    L = 10
    npoints = 101
    xvals, dx = util.create_xvals(L, npoints, endpoint=False)
    kvals = np.fft.fftfreq(npoints, d=dx) * 2 * np.pi

    def test_FT_iFT_identity(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        phi = util.FT(psi, self.xvals, self.kvals)
        psi_reconstructed = util.iFT(phi, self.xvals, self.kvals)
        assert np.allclose(psi, psi_reconstructed)


    def test_FT_assertion(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1)
        with np.testing.assert_raises(AssertionError):
            util.FT(psi, self.xvals, self.kvals[:-1]) # mismatch in length of k

    def test_iFT_assertion(self):
        phi = util.FT(util.gaussian_wave_packet(self.xvals, x0=0, sigma=1, p0=1), self.xvals, self.kvals)
        with np.testing.assert_raises(AssertionError):
            util.iFT(phi, self.xvals[:-1], self.kvals) # mismatch in length of x

class Test_gaussian_wave_packet:

    L = 20
    npoints = 201
    xvals, dx = util.create_xvals(L, npoints)

    def test_gaussian_wave_packet_normalization(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=1)
        norm = np.sum(np.abs(psi)**2) * self.dx
        assert np.isclose(norm, 1)

    def test_gaussian_wave_packet_symmetry(self):
        psi = util.gaussian_wave_packet(self.xvals, x0=-3, sigma=1, p0=0)
        assert np.allclose(psi[60], psi[80]) # check that the wave packet is symmetric around x0=-3

class Test_create_tvecs:

    def test_create_tvecs_length(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert len(tvec) == tsteps + 1

    def test_create_tvecs_values(self):
        tsteps = 10
        dt = 0.1
        tvec = util.create_tvecs(tsteps, dt)
        assert np.isclose(tvec[0], 0)
        assert np.isclose(tvec[-1], tsteps * dt)

class Test_idx2state_state2idx:

    N1 = 3
    N2 = 4

    def test_idx2state_state2idx_consistency(self):
        idx_recon = []
        for i in range(self.N1 * self.N2):
            state = util.idx2state(self.N1, self.N2, i)
            idx_recon.append(util.state2idx(self.N1, self.N2, state))
        assert np.array_equal(np.arange(self.N1 * self.N2), idx_recon)

    def test_idx2state_state2idx_out_of_bounds(self):
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, -1) # negative index
        with np.testing.assert_raises(AssertionError):
            util.idx2state(self.N1, self.N2, self.N1 * self.N2) # index equal to dimension
        idx = util.state2idx(self.N1, self.N2, [self.N1, 0]) # n1 out of bounds
        assert idx == -1
        idx = util.state2idx(self.N1, self.N2, [0, self.N2]) # n2 out of bounds
        assert idx == -1

    def test_idx2state_specific_cases(self):
        # test some specific cases for idx2state and state2idx
        i = 0
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [0, 0]
        i = 5
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [1, 1]
        i = 11
        state = util.idx2state(self.N1, self.N2, i)
        assert state == [2, 3]


###################### Solution sheet 4 ######################


class Test_create_coherent_state:

    N = 100

    def test_create_coherent_state_normalization(self):
        alpha = 1 + 1j
        state = util.create_coherent_state(self.N, alpha)
        norm = np.sum(np.abs(state)**2)
        assert np.isclose(norm, 1)

    def test_create_coherent_state_alpha_zero(self):
        alpha = 0
        state = util.create_coherent_state(self.N, alpha)
        expected = np.zeros(self.N)
        expected[0] = 1
        assert np.allclose(state, expected)

    def test_a_operator_sparse_consistency(self):
        alpha = 3
        a_op = ops.a_operator_sparse(self.N).toarray()
        init_state = util.create_coherent_state(self.N, alpha)
        applied_a_op = a_op @ init_state
        applied_a_op /= alpha # should be equal to the original state
        assert np.allclose(1, np.vdot(applied_a_op[:-1], init_state[:-1]), atol=1e-10)
        # check the coherent-state eigenvalue relation for the annihilation operator, up to truncation effects in the last basis element.

class Test_expectation_value:

    N = 100

    L = 20
    npoints = 2001
    xvals, dx = util.create_xvals(L, npoints)

    def test_expectation_value_hermitian(self):
        # test that the expectation value of a Hermitian operator is real
        alpha = 1 + 1j
        state = util.create_coherent_state(self.N, alpha)
        x_op = ops.x_operator_sparse(self.N).toarray()
        exp_val = util.expectation_value(state, x_op)
        assert np.isclose(np.imag(exp_val), 0)

    def test_expectation_value_known(self):
        # test the expectation value of the number operator in a coherent state, which should be |alpha|^2
        alpha = 2.0 + 1j
        state = util.create_coherent_state(self.N, alpha)
        print("state: ", state)
        n_op = ops.n_operator_sparse(self.N)
        exp_val = util.expectation_value(state, n_op)
        expected = np.abs(alpha)**2
        assert np.isclose(exp_val, expected, atol=1e-10)

    def test_expectation_value_iterable(self):
        # test that the function can handle an iterable of operators
        x0 = -1
        sigma = 1
        p0 = 1
        state = util.gaussian_wave_packet(self.xvals, x0=x0, sigma=sigma, p0=p0)
        print("x_prob = ", sum(np.abs(state)**2 * self.xvals) * self.dx)
        
        
        x_op = np.diag(self.xvals) # position operator in the x basis
        p_op = np.zeros((self.npoints, self.npoints), dtype=complex) # momentum operator in the x basis, using finite difference approximation
        for i in range(1, self.npoints - 1):
            p_op[i, i - 1] = 1j / (2 * self.dx)
            p_op[i, i + 1] = -1j / (2 * self.dx)

        exp_vals = util.expectation_value(state, [x_op, p_op])
        assert len(exp_vals) == 2
        assert np.isclose(np.imag(exp_vals[0]), 0) # expectation value of x should be real
        assert np.isclose(np.imag(exp_vals[1]), 0) # expectation value of p should be real
        expected = [x0, p0]
        exp_val_norm = np.real(exp_vals) * self.dx
        assert np.allclose(exp_val_norm, expected, atol=1e-4)


###################### Exercise sheet 7 ######################


class Test_Husimi_proj:

    N = 100
    #phi_test = np.pi / 3
    #theta_test = np.pi / 2
    ngrid = 101

    def test_husimi_front_back_symmetry(self):
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi / 2

        #psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_back
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_front_back_symmetry_theta_pi(self):
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi

        #psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_back
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_top_front_symmetry(self):
        # test that the Husimi functions for the top and front states are symmetric with respect to the theta axis
        # test that the Husimi functions for the front and back states are symmetric with respect to the phi axis

        phi_test = np.pi / 3
        theta_test = np.pi / 2

        psi_top = util.CSS(self.N, phi_test, theta_test)  # top state of the CSS basis
        psi_front = util.CSS(self.N, theta_test, phi_test)  # front state of the CSS basis
        #psi_back = util.CSS(self.N, theta_test, np.pi - phi_test)  # back state of the CSS basis

        Z, Y, H_front = util.Husimi_front(self.N, psi_front, self.ngrid, self.ngrid)
        Z, Y, H_top = util.Husimi_top(self.N, psi_top, self.ngrid, self.ngrid)

        #Z, Y, H_back = util.Husimi_back(self.N, psi_back, self.ngrid, self.ngrid)
        diff = H_front - H_top
        assert np.allclose(diff, 0, atol=1e-10)

    def test_husimi_th_phi_symmetry(self):
        # test that the Husimi functions are symmetric with respect to theta -> pi - theta
        phi = np.pi / 3
        theta_1 = np.pi / 3
        theta_2 = np.pi - theta_1

        psi_1 = util.CSS(self.N, theta_1, phi)  # state of the CSS basis
        psi_2 = util.CSS(self.N, theta_2, phi)  # state of the CSS basis

        Theta, Phi, H1 = util.Husimi_th_ph(self.N, psi_1, self.ngrid, self.ngrid)
        Theta, Phi, H2 = util.Husimi_th_ph(self.N, psi_2, self.ngrid, self.ngrid)
        diff = H1 - np.flip(H2, axis=0) # flip H2 along the theta axis
        assert np.allclose(diff, 0, atol=1e-10)
    

    def test_husimi_z_phi_symmetry_phi(self):
        # test that the Husimi functions are symmetric with respect to z -> -z

        phi = np.pi / 3
        theta_1 = np.pi / 3
        theta_2 = np.pi - theta_1

        psi_1 = util.CSS(self.N, theta_1, phi)  # state of the CSS basis
        psi_2 = util.CSS(self.N, theta_2, phi)  # state of the CSS basis

        Z, Phi, H1 = util.Husimi_z_phi(self.N, psi_1, self.ngrid, self.ngrid)
        Z, Phi, H2 = util.Husimi_z_phi(self.N, psi_2, self.ngrid, self.ngrid)
        diff = H1 - np.flip(H2, axis=0) # flip H2 along the z axis
        assert np.allclose(diff, 0, atol=1e-10)
    

###################### Exercise sheet 8 ######################


class Test_partial_trace:

    def test_partial_trace_product(self):
        N = 3
        psi_full = np.eye(1, 2 ** N, 5)[0] # |101> state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_full, 1) # trace out the last spin
        expected_psi = np.eye(1, 2 ** (N - 1), 2)[0] # |10> state in the reduced Hilbert space
        expected_rho = np.outer(expected_psi, expected_psi.conj())
        assert np.allclose(rho_reduced, expected_rho)

    def test_partial_trace_entangled(self):
        N = 3
        psi_ghz = (1 / np.sqrt(2)) * (np.eye(1, 2 ** N, 0)[0] + np.eye(1, 2 ** N, 7)[0]) # GHZ state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_ghz, 2) # trace out the last two spins
        expected_rho = 0.5 * np.eye(2) # reduced density matrix for the first spin, which is maximally mixed
        assert np.allclose(rho_reduced, expected_rho)

class Test_entanglement_entropy:

    def test_entanglement_entropy_product(self):
        N = 3
        psi_full = np.eye(1, 2 ** N, 5)[0] # |101> state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_full, 1) # trace out the last spin
        S = util.entanglement_entropy(rho_reduced) # trace out the last spin
        expected_S = 0.0 # product state should have zero entanglement entropy
        assert np.isclose(S, expected_S)

    def test_entanglement_entropy_entangled(self):
        N = 3
        psi_ghz = (1 / np.sqrt(2)) * (np.eye(1, 2 ** N, 0)[0] + np.eye(1, 2 ** N, 7)[0]) # GHZ state in the full Hilbert space
        rho_reduced = util.partial_trace(psi_ghz, 1) # trace out the last two spins
        S = util.entanglement_entropy(rho_reduced) # trace out the last two spins
        expected_S = 1 # reduced density matrix for the first spin is maximally mixed, so S should be log(2)
        assert np.isclose(S, expected_S)
        
    def test_entanglement_entropy_mixed_state(self):
        # test that the entanglement entropy of a mixed state is non-negative
        rho = np.diag([1/3, 2/3]) # mixed state for a single qubit
        S = util.entanglement_entropy(rho)
        S_expected = 1/3 * np.log2(3) +  2/3 * np.log2(3 / 2)
        assert np.isclose(S, S_expected)


###################### Exercise sheet 9 ######################

class Test_n_party_idx2state:

    def test_n_party_idx2state_first_state(self):
        # Test edge cases for n_party_idx2state
        N = 6
        local_dim = 3
        
        idx = 0
        expected_state = [-1] * N
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)
    
    def test_n_party_idx2state_last_state(self):
        N = 6
        local_dim = 3
        
        idx = local_dim ** N - 1
        expected_state = [1] * N
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)

    def test_n_party_idx2state_middle_state(self):
        N = 6
        local_dim = 3
        
        idx = 11 # corresponds to state [-1, -1, -1, 0, -1, 1]
        expected_state = [-1, -1, -1, 0, -1, 1]
        state = util.n_party_idx2state(idx, local_dim, N)
        assert np.allclose(state, expected_state)

