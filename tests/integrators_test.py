import numpy as np
import numpy.linalg as LA
import pytest
#import scipy.sparse as sparse
from scipy.integrate import ode


import Comp_Quant_Dynam.hamiltonians as hams
import Comp_Quant_Dynam.utility as util
import Comp_Quant_Dynam.unitaries as unitaries
import Comp_Quant_Dynam.operators as ops
import Comp_Quant_Dynam.integrators as integrators


###################### Exercise sheet 6 ######################

class Test_integrate_ODE:
    
    def test_integrate_ODE_ED_Euler(self):
        N = 4
        state = np.eye(1, 2**N, 6).flatten() # initial state |0110>
        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_y = ops.sigma_y_sparse() # single-site sigma_y operator
        
        local_dims = [2] * N
        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x) + 0.5 * ops.n_party_op_sparse(local_dims, 2, sig_y)
        
        t_steps = 100
        t_end = 2 * np.pi
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)

        obsv_vec = [ops.n_party_op_sparse(local_dims, 3, ops.sigma_z_sparse())] # observable: sigma_z on site 3

        observables_ED = unitaries.calc_expv_ED(obsv_vec, H, state, tvec)
        stepper_func = integrators.Euler_step
        int_steps_per_dtout = 500
        stepper_args = None
        observables_int = integrators.integrate_ODE(stepper_func, obsv_vec, H, state, tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(observables_ED - observables_int)))
        assert np.allclose(observables_ED, observables_int, atol = 1e-3)

    def test_integrate_ODE_warnings_Euler(self):
        N = 4
        state = np.eye(1, 2**N, 8).flatten() # initial state |1000>
        sig_x = ops.sigma_x_sparse() # single-site sigma_x operator
        sig_z = ops.sigma_z_sparse() # single-site sigma_z operator

        local_dims = [2] * N
        non_hermitian_op = sig_x + 1j * sig_z # non-Hermitian operator
        nh3 = ops.n_party_op_sparse(local_dims, 3, non_hermitian_op) # non-Hermitian operator acting on site 3
        
        H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x)
        
        t_steps = 100
        t_end = 2 * np.pi
        tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)
    
        stepper_func = integrators.Euler_step
        obsv_vec = [nh3] # non-Hermitian observable acting on site 3
        pytest.warns(UserWarning, integrators.integrate_ODE, stepper_func, obsv_vec, H, state, tvec, int_steps_per_dtout=10, stepper_args=None)

        
###################### Solution sheet 6 ######################


class Test_integrate_ODE_contd:

    N = 4

    @classmethod
    def setup_class(cls):
        cls.state = np.eye(1, 2**cls.N, 6).flatten()  # initial state |0110>
        sig_x = ops.sigma_x_sparse()  # single-site sigma_x operator
        sig_y = ops.sigma_y_sparse()  # single-site sigma_y operator

        local_dims = [2] * cls.N
        cls.H = 0.5 * ops.n_party_op_sparse(local_dims, 3, sig_x) + 0.5 * ops.n_party_op_sparse(local_dims, 2, sig_y)

        t_steps = 100
        t_end = 10
        cls.tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)

        cls.obsv_vec = [ops.n_party_op_sparse(local_dims, 3, ops.sigma_z_sparse())]  # observable: sigma_z on site 3
        cls.observables_ED = unitaries.calc_expv_ED(cls.obsv_vec, cls.H, cls.state, cls.tvec)
    
    def test_integrate_ODE_ED_RK2(self):
        stepper_func = integrators.RK2_step
        int_steps_per_dtout = 100
        stepper_args = None
        observables_int = integrators.integrate_ODE(stepper_func, self.obsv_vec, self.H, self.state, self.tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(self.observables_ED - observables_int)))
        assert np.allclose(self.observables_ED, observables_int, atol = 1e-6)

    def test_integrate_ODE_ED_RKn(self):
        stepper_func = integrators.RKn_step
        int_steps_per_dtout = 2
        n = 10
        stepper_args = [n]
        observables_int = integrators.integrate_ODE(stepper_func, self.obsv_vec, self.H, self.state, self.tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(self.observables_ED - observables_int)))
        assert np.allclose(self.observables_ED, observables_int, atol = 1e-6)

    def test_integrate_ODE_ED_CN(self):
        stepper_func = integrators.CN_step
        int_steps_per_dtout = 100
        stepper_args = None
        observables_int = integrators.integrate_ODE(stepper_func, self.obsv_vec, self.H, self.state, self.tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(self.observables_ED - observables_int)))
        assert np.allclose(self.observables_ED, observables_int, atol = 1e-6)

    def test_integrate_ODE_ED_Arnoldi(self):
        stepper_func = integrators.Arnoldi_step
        int_steps_per_dtout = 5
        krylov_dim = 3
        stepper_args = [krylov_dim]
        observables_int = integrators.integrate_ODE(stepper_func, self.obsv_vec, self.H, self.state, self.tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(self.observables_ED - observables_int)))
        assert np.allclose(self.observables_ED, observables_int, atol = 1e-6)

    def test_integrate_ODE_ED_scipyODE(self):
        stepper_func = integrators.scipyODE_step
        int_steps_per_dtout = 1
        r = ode(integrators.schroedinger_diff_eq).set_integrator('zvode', method='adams', with_jacobian=False, rtol=1e-10, atol=1e-12)
        r.set_initial_value(self.state, 0).set_f_params(self.H)
        stepper_args = [r]
        observables_int = integrators.integrate_ODE(stepper_func, self.obsv_vec, self.H, self.state, self.tvec, int_steps_per_dtout, stepper_args)
        print("Deviation between ED and integration results:", np.max(np.abs(self.observables_ED - observables_int)))
        assert np.allclose(self.observables_ED, observables_int, atol = 1e-6)


###################### Solution sheet 7 ######################


class Test_TWA_ED:

    N = 50
    omega = 0.3

    @classmethod
    def setup_class(cls):
        cls.state = util.CSS(cls.N, 0, 0)

        H_mat = hams.build_H_TFIM(cls.N, cls.omega)
        evals, evecs = LA.eigh(H_mat.toarray())
        cls.iniProj = unitaries.init_coeffs_eigenbasis(cls.state, evecs)

        Sx, Sy, Sz = ops.build_spin_ops_sparse(cls.N)
        cls.spin_op_vec = np.array([Sx, Sy, Sz]) / (cls.N / 2)
        local_dims = [2] * cls.N

        t_steps = 100
        t_end = 10
        cls.tvec = util.create_tvecs(tsteps=t_steps, dt=t_end / t_steps)
        cls.obs_exact = np.zeros((len(cls.tvec), 3), dtype=float)
        for t_idx in range(len(cls.tvec)):
            Psit = unitaries.t_evol_eigenbasis(cls.iniProj, cls.tvec[t_idx], evals, evecs)
            cls.obs_exact[t_idx, :] = np.real(util.expectation_value(Psit, cls.spin_op_vec)) 

    def test_TWA_ED(self):

        np.random.seed(8675309) # Jenny Jenny, can't you see? --- set seed for reproducibility
        n_samples = 500
        xy_ini = np.random.normal(0, 1 / np.sqrt(self.N), (n_samples, 2)) # Gaussian random numbers for x and y
        ini_list = np.transpose([xy_ini[:, 0], xy_ini[:, 1], np.full(n_samples, 1)]) # add constant 1 for z

        all_trajectories = np.zeros((n_samples, len(self.tvec), 3))

        # loop over initial conditions
        for i in range(n_samples):
            all_trajectories[i] = integrators.get_trajectory(ini_list[i], self.tvec, self.omega)

        obs_TWA = np.mean(all_trajectories, axis=0)

        assert np.allclose(self.obs_exact, obs_TWA, atol=1e-2)