import numpy as np   # standard numerics library
import numpy.linalg as LA
from scipy import sparse
import scipy.linalg as sciLA
import scipy.sparse.linalg as sLA
from scipy.integrate import ode
import time
import warnings

from Comp_Quant_Dynam.utility import expectation_value
from Comp_Quant_Dynam.unitaries import calc_expv_ED


###################### Exercise sheet 6 ######################

def integrate_ODE(stepper_func, obsv_vec, H_mat, ini, tvec_out, int_steps_per_dtout, stepper_args):
    """
    Integrates the ODE `dy/dt = -i H_mat @ y` using the integrator defined by `stepper_func` and returns the expectation values of the observables provided in `obsv_vec` at the time steps defined in `tvec_out`.
    The initial state is given by `ini`, and the Hamiltonian is given by `H_mat`.
    The time step size for the integrator is given by `dt = (tvec_out[1] - tvec_out[0]) / int_steps_per_dtout`, and the number of integration steps per output time step is given by `int_steps_per_dtout`.
    Optional arguments for the integrator can be provided in `stepper_args`.
    """
    
    n_obsv = len(obsv_vec)

    dt_out = tvec_out[1] - tvec_out[0] # time step size for the output time steps
    dt_int = dt_out / int_steps_per_dtout # time step size for the integrator

    tvec_int = np.linspace(0, tvec_out[-1], int((len(tvec_out) - 1) * int_steps_per_dtout) + 1) # time vector for the integrator

    n_t_out = len(tvec_out) # number of output time steps
    n_t_int = len(tvec_int) # number of integration time steps

    observables = np.zeros((n_obsv, n_t_out), dtype=float) # container for observables

    t1 = time.time() # measure time

    Psit = ini.copy() # initialize wave function

    # store initial values of observables
    observables[:, 0] = np.real(expectation_value(Psit, obsv_vec))
    # Integration steps:
    for idx_t in range(1, n_t_int):
        Psit = stepper_func(Psit, H_mat, dt_int, stepper_args)
        if idx_t % int_steps_per_dtout == 0:
            # calculate and store observables:
            idx_out = int(idx_t / int_steps_per_dtout)
            exp_vals = expectation_value(Psit, obsv_vec)
            if not np.allclose(np.imag(exp_vals), 0.0):
                warnings.warn("Some observables have non-zero imaginary parts")
            observables[:, idx_out] = np.real(exp_vals)

    t2=time.time() # end run time measurement
    print('time for integration was ' + str(t2 - t1))

    return observables


def schroedinger_diff_eq(t, y, H_mat):
    """
    Returns the right-hand side of the time-independent Schrödinger equation `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat`.
    This function can be used with ODE integrators that require a function of this form.
    The argument `t` is included for compatibility with ODE integrators, but it is not used in this function since the Hamiltonian is time-independent.
    """
    return -1j * H_mat @ y

def Euler_step(y, H_mat, dt, stepper_args):
    """
    Returns the next step of the Euler method for integrating the Schrödinger equation `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    """

    increment = schroedinger_diff_eq(0, y, H_mat) # calculate the increment using the right-hand side of the Schrödinger equation

    # things to think about:
    # how efficient is @ for sparse matrices? Is there a more efficient method?
    # improvement by using the Jacobian?
    
    return y + dt * increment

def loop_time_step(stepper_func, obsv_vec, H_mat, ini, tvec, red_factor, n_red_step, stepper_args):
    """
    Returns the deviations of the observables provided in `obsv_vec` calculated with the integrator defined by `stepper_func` from the exact diagonalization results for different time step sizes.
    The time step size is reduced by a factor of `red_factor` in each iteration, and the number of iterations is given by `n_red_step`.
    The initial state is given by `ini`, and the Hamiltonian is given by `H_mat`.
    Optional arguments for the integrator can be provided in `stepper_args`.
    """

    n_obsv = len(obsv_vec)

    dt_out = tvec[1] - tvec[0] # output time step size

    deviations = np.zeros((n_red_step, n_obsv, len(tvec))) # store only the deviations between numerical integration and ED.

    int_steps_per_dtout = 1

    observables_ED = calc_expv_ED(obsv_vec, H_mat, ini, tvec) # calculate observables with exact diagonalization for comparison

    step_sizes = np.zeros((n_red_step)) # vector of time step sizes to try.

    # decrease the time step in each iteration
    for idx_stepsize in range(n_red_step):
    
        # store current integration step size
        step_sizes[idx_stepsize] = dt_out / int_steps_per_dtout
        print("dt_integrator = ", step_sizes[idx_stepsize])

        # integration
        observables_Integrator = integrate_ODE(stepper_func, obsv_vec, H_mat, ini, tvec, int_steps_per_dtout, stepper_args)

        # store the deviations between the integrator and ED results for the current time step size
        deviations[idx_stepsize, :, :] = observables_Integrator - observables_ED

        # increase the number of integration steps per output time step by the reduction factor
        int_steps_per_dtout = int_steps_per_dtout * red_factor 

    return np.abs(deviations), step_sizes

def generate_krylov_subspace(N, n, y, H_mat):
    """
    Generates the Krylov subspace of degree `n` for a system of size `N` starting from the vector `y` and using the Hamiltonian `H_mat`.
    Returns the matrix of Krylov vectors `Qs` and the tridiagonal h-matrix `h` that represents the Hamiltonian in the Krylov subspace.
    The Krylov vectors are generated using the Lanczos algorithm, and the h-matrix is generated as a dense matrix.
    If the Krylov subspace is smaller than `n` (i.e., if we encounter an eigenstate of the Hamiltonian), the function terminates early and returns the Krylov vectors and h-matrix generated up to that point.
    """
    # note: h is generated as a dense matrix, making it sparse may improve performance
    
    Qs = np.zeros((N + 1, n + 1), dtype = complex) # dimension of Krylov subspace is n+1
    h = np.zeros((n + 1, n + 1), dtype = complex)
    Qs[:, 0] = y / LA.norm(y) # normalize just in case
    for i in range(1,n+1):
        v = H_mat @ Qs[:, i-1] # apply H to previous Krylov vector
        if i > 1:
            # calculate h matrix elements:
            hcol = Qs[:, (i - 2) : i].conj().T @ v
            h[(i - 2) : i, i - 1] = hcol
            h[i - 1, i - 2] = hcol[0].conj()
            # subtract projections on previous Krylov vectors (Gram Schmidt):
            Qs[:,i] = v - np.sum(Qs[:, (i - 2) : i] * hcol.T, axis=-1)
        else:
            # for i=1, there is only one previous Krylov vector
            hcol = Qs[:, (i - 1) : i].conj().T @ v
            h[(i - 1) : i, i - 1] = hcol
            Qs[:, i] = v - np.sum(Qs[:, (i - 1) : i] * hcol.T, axis=-1)
        # normalize and store new Krylov vector
        norm = LA.norm(Qs[:, i])
        if norm < 10e-10: # If we already have an eigenstate, terminate. Rest of Qs and h can stay zero.
            return Qs, h
        Qs[:, i] /= norm
    # last row and column of h
    v = H_mat @ Qs[:, n]
    hcol = Qs[:, (n - 1) : (n + 1)].conj().T @ v
    h[(n - 1) : (n + 1), n] = hcol
    h[n, n - 1] = hcol[0].conj()
    return Qs, h


###################### Solution sheet 6 ######################
def RK2_step(y, H_mat, dt, stepper_args):
    """
    Returns the next step of the second-order Runge-Kutta method for integrating the Schrödinger equation
    `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    """

    Hy = H_mat @ y
    return y - 1j * dt * Hy - dt ** 2 / 2 * H_mat @ Hy

def RKn_step(y, H_mat, dt, stepper_args):
    """
    Returns the next step of the n-th order Runge-Kutta method for integrating the Schrödinger equation
    `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    The order `n` is provided in the `stepper_args`.
    """

    n = int(stepper_args[0]) # order of the Runge-Kutta method
    y_j = yout = y
    for i in range(1, n + 1):
        y_j =  dt / i * schroedinger_diff_eq(0, y_j, H_mat)
        yout = yout + y_j
    return yout

def CN_step(y, H_mat, dt, stepper_args):
    """
    Returns the next step of the Crank-Nicolson method for integrating the Schrödinger equation
    `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    Ideally, we would like to precompute the A and B matrices for the Crank-Nicolson method, but since we want to use this function with the loop_time_step function, we need to compute them in each step.
    If we wanted to avoid this, we would need to change the loop_time_step function to allow for precomputation of these matrices.
    """

    dim = H_mat.shape[0]
    A = sparse.csr_array(sparse.eye(dim) + 1j * dt / 2 * H_mat)
    B = sparse.csr_array(sparse.eye(dim) - 1j * dt / 2 * H_mat)
    RHS = B @ y
    return sLA.spsolve(A, RHS)

def Arnoldi_step(y, H_mat, dt, stepper_args):
    """
    Returns the next step of the Arnoldi method for integrating the Schrödinger equation
    `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    The order `n` of the Arnoldi method (i.e., the dimension of the Krylov subspace) is provided in the `stepper_args`.
    """
    n = int(stepper_args[0]) # order of the Arnoldi method, i.e., dimension of the Krylov subspace is n+1
    N = len(y) - 1 # dimension of the system is N+1
    Qs, h = generate_krylov_subspace(N, n, y, H_mat)
    e1 = np.eye(n + 1)[0]
    return Qs @ (sciLA.expm(-1j * dt * h) @ e1)

def scipyODE_step(y, H_mat, dt,stepper_args):
    """Returns the next step of the ODE integrator defined by `stepper_args` for integrating the Schrödinger equation
    `dy/dt = -i H_mat @ y` for a state vector `y` and Hamiltonian `H_mat` with time step size `dt`.
    The scipy ODE integrator is provided in `stepper_args` and is assumed to be already initialized with the initial value and the Hamiltonian as a parameter.
    """
    r = stepper_args[0] # r is the ODE integrator object that has already been initialized with the initial value and the Hamiltonian as a parameter
    r.integrate(r.t + dt)
    return r.y


###################### Solution sheet 7 ######################

def TFIM_MF_diff_eq(t, y, omega):
    """
    Returns the right-hand side of the mean-field equations for the transverse-field Ising model (TFIM) in terms of the spin components `y` and the transverse field strength `omega`.
    The mean-field equations are given by:
    dy[0]/dt = y[1] * y[2]
    dy[1]/dt = -y[0] * y[2] + omega * y[2]
    dy[2]/dt = -omega * y[1]
    where y[0], y[1], and y[2] correspond to the spin components Sx, Sy, and Sz, respectively.
    The argument `t` is included for compatibility with ODE integrators, but it is not used in this function since the equations are time-independent.
    """
    eqs = [
        y[1] * y[2],
        -y[0] * y[2] + omega * y[2],
        -omega * y[1]
    ]
    return eqs

def get_trajectory(y0, tvec, ome):
    """
    Returns the trajectory of the spin components for the mean-field equations of the transverse-field Ising model (TFIM) starting from the initial condition `y0`, evaluated at the time steps defined in `tvec`, and using the transverse field strength `ome`.
    The trajectory is calculated using the ODE integrator from scipy, and the right-hand side of the equations is given by the `TFIM_MF_diff_eq` function.
    """

    r = ode(TFIM_MF_diff_eq).set_integrator(
        "vode",
        method="adams",
        with_jacobian=False,
        rtol=1e-10,
        atol=1e-12,
    )
    r.set_initial_value(np.asarray(y0, dtype=float), 0.0).set_f_params(ome)

    t1 = float(tvec[-1])
    dt = float(tvec[1] - tvec[0])
    trajectory = np.zeros((len(tvec), 3), dtype=float)

    trajectory[0] = y0
    i = 1
    while r.successful() and r.t < t1 and i < len(tvec):
        r.integrate(r.t + dt)
        trajectory[i] = r.y
        i += 1
    return trajectory