import numpy as np   # standard numerics library
import numpy.linalg as LA
import time
import warnings

from Comp_Quant_Dynam.utility import expectation_value, _check_if_sized


###################### Solution sheet 2 ######################

def t_evol_eigenbasis(init_coeffs, t, evals, evecs):
    """
    Returns the state in the computational basis at time 't' given the initial state coefficients 'init_coeffs'
    in the eigenbasis for a Hamiltonian with eigenvalues 'evals' and eigenvectors 'evecs'.
    """

    phase_factors = np.exp(-1j * evals * t) # compute the phase factors for each eigenstate
    evolved_eigen_coeffs = init_coeffs * phase_factors
    return evecs @ evolved_eigen_coeffs

def init_coeffs_eigenbasis(psi0, evecs):
    """
    Returns the coefficients of the initial state 'psi0' in the eigenbasis given by 'evecs'.
    """

    return np.conjugate(evecs.T) @ psi0


###################### Solution sheet 3 ######################

def t_evol_split_step_fourier(psi0, V_func, tvec, xvals):
    """
    Returns the time evolution of the state 'psi0' under the potential 'V_func' using the split-step Fourier method for time evolution.
    V_func is a function that takes time 't' as input and returns the potential at that time.
    The time vector is given by 'tvec', and the position and momentum grids are given by 'x' and 'k', respectively.
    """

    npoints = len(xvals) # number of grid points
    dx = xvals[1] - xvals[0] # grid spacing
    kvals = 2 * np.pi * np.fft.fftfreq(npoints, d=dx)
    tsteps = len(tvec) - 1 # number of time steps
    dt = tvec[1] - tvec[0] # time step size

    # momentum grid corresponding to the position grid
    
    # container for storing the result
    psit = np.zeros((len(tvec), npoints), dtype=complex)

    # store initial value
    psit[0] = psi0

    for i in range(tsteps):
        
        # apply potential
        psit[i + 1, :] = np.exp(-1j * dt * V_func(tvec[i])) * psit[i, :]
        # go to Fourier space
        psit[i + 1, :] = np.fft.fft(psit[i + 1, :])
        # apply kinetic part
        psit[i + 1, :] = np.exp(-1j * dt * kvals**2 / 2) * psit[i + 1, :]
        # go back to real space
        psit[i + 1, :] = np.fft.ifft(psit[i + 1, :])
        # store the result
        psit[i + 1] = psit[i + 1, :]

    return psit


###################### Exercise sheet 6 ######################

def calc_expv_ED(obsv_vec, H_mat, ini, tvec):
    """
    Returns the expectation values of the observables in `obsv_vec` at the times given by `tvec` for a system with Hamiltonian `H_mat`
    and initial state `ini` using exact diagonalization.
    """

    n_obsv, obsv_vec = _check_if_sized(obsv_vec)
    observables = np.zeros((n_obsv, len(tvec)), dtype=float) # container for results

    # ED solution
    t1=time.time()
    evals, evecs = LA.eigh(H_mat.toarray()) # diagonalize
    # calculate projections on eigenstates
    iniProj = init_coeffs_eigenbasis(ini, evecs) 
    for t_idx, t in enumerate(tvec):
        Psit = t_evol_eigenbasis(iniProj, t, evals, evecs)
        exp_vals = expectation_value(Psit, obsv_vec)
        if not np.allclose(np.imag(exp_vals), 0.0):
            warnings.warn("Some observables have non-zero imaginary parts")
        observables[:, t_idx] = np.real(exp_vals)
    t2=time.time()
    
    print('time for ED was '+str(t2 - t1))

    return observables