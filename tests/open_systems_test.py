import numpy as np
#import numpy.linalg as LA
#from scipy.sparse.linalg import eigsh
from scipy.integrate import ode
#from math import comb

import pytest

import Comp_Quant_Dynam.open_systems as open_systems
import Comp_Quant_Dynam.utility as utility
import Comp_Quant_Dynam.integrators as integrators


###################### Solution sheet 11 ######################


class Test_EIT_ME:

    params = {
        "omegaP": 1.3,
        "omegaC": 2.7,
        "gammaP": 2.0,
        "gammaC": 0.0,
        "Gamma": 0.0,
        "DeltaP": 0.0,
        "DeltaC": 0.0
    }

    L_mat = open_systems.build_L_mat_EIT(params)
    rho_ss = open_systems.rho_ss(L_mat)


    def test_steady_state(self):
        """
        Test that the steady state is the dark state for standard EIT parameters.
        """

        omegaP = self.params["omegaP"]
        omegaC = self.params["omegaC"]
        norm = np.sqrt(omegaP ** 2 + omegaC ** 2)

        dark_state = np.array([omegaC / norm, -omegaP / norm, 0], dtype='complex')
        rho_dark = np.outer(dark_state, dark_state.conj())
        # Check that rho_ss is close to the projector onto the dark state
        assert np.allclose(self.rho_ss, rho_dark, atol=1e-10)

    def test_ME_steady_state(self):

        ME_RHS = self.L_mat @ self.rho_ss.flatten()
        # Check that the right-hand side of the master equation is close to zero
        assert np.allclose(ME_RHS, np.zeros_like(ME_RHS), atol=1e-10)

class Test_EIT_MCWF:

    params = {
        "omegaP": 1.3,
        "omegaC": 2.7,
        "gammaP": 2.0,
        "gammaC": 0.0,
        "Gamma": 0.0,
        "DeltaP": 0.0,
        "DeltaC": 0.0
    }

    H, L_list = open_systems.build_EIT_operators(params)
    H_non_herm = open_systems.build_H_non_herm(H, L_list)

    L_mat = open_systems.build_L_mat_EIT(params)
    rho_ss = open_systems.rho_ss(L_mat)

     # time grid
    tend = 30
    dt = .1
    tsteps = int(tend / dt)
    tvec = utility.create_tvecs(tsteps, dt)

    ini = np.array([1, 0, 0], dtype='complex')

    n_traj = 250

    all_trajectories = np.zeros((n_traj, len(tvec), len(H_non_herm)), dtype='complex')

    np.random.seed(8675309)  # Jenny don't change your number!

    for traj_idx in range(n_traj):
        all_trajectories[traj_idx] = open_systems.get_trajectory(H_non_herm, L_list, ini, tvec)

    def test_MCWF_steady_state(self):

        # Compute the average density matrix from the trajectories
        rho_ss_MCWF = np.mean([np.outer(traj[-1], traj[-1].conj()) for traj in self.all_trajectories], axis=0)
        
        omegaP = self.params["omegaP"]
        omegaC = self.params["omegaC"]
        norm = np.sqrt(omegaP ** 2 + omegaC ** 2)

        dark_state = np.array([omegaC / norm, -omegaP / norm, 0], dtype='complex')
        rho_dark = np.outer(dark_state, dark_state.conj())
        # Check that rho_ss is close to the projector onto the dark state

        assert np.allclose(rho_ss_MCWF, rho_dark, atol=1e-6)

    def test_relaxation_ss(self):
       
        # initial condition
        y0 = utility.pij(3, 0, 0).reshape(9,) # all in g1
        t0 = 0

        r = ode(integrators.linblad_master_eq).set_integrator('zvode', method='adams', with_jacobian=False)
        r.set_initial_value(y0, t0).set_f_params(self.L_mat)

        rho_t = np.zeros((len(self.tvec), 3, 3),dtype='complex')
        
        rho_t[0] = y0.reshape((3, 3))

        # integration
        i=1
        while r.successful() and r.t < self.tend and i < len(self.tvec):
            r.integrate(r.t + self.dt)
            rho = r.y
            rho_t[i] = np.reshape(rho, (3, 3))
            i += 1

        # Vectorized computation of mean density matrix from trajectories:
        # all_trajectories has shape (n_traj, tsteps, dim)
        # einsum computes for each time t the sum over trajectories of outer products
        rho_MCWF = np.einsum('mtn,mts->tns', self.all_trajectories, self.all_trajectories.conj()) / float(self.n_traj)

        assert np.allclose(rho_t, rho_MCWF, atol=5e-2)

    def test_build_EIT_operators_ladder_scheme(self):
        """
        Test that the build_EIT_operators function returns the correct Hamiltonian and Lindblad operators for a three-level ladder system.
        """

        H_ladder, L_list_ladder = open_systems.build_EIT_operators_ladder_scheme(self.params)

        # Check that the Hamiltonian is Hermitian
        assert np.allclose(H_ladder, H_ladder.conj().T)

        assert np.allclose(H_ladder, self.H)
    
        assert np.allclose(L_list_ladder[0].T, self.L_list[0])

        assert np.allclose(L_list_ladder[1], self.L_list[1])
