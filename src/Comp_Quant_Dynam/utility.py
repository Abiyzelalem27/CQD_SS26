


import numpy as np   # standard numerics library
from numpy import linalg as LA
from scipy.linalg import expm
from collections.abc import Iterable, Sequence
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan, sqrt, exp
from scipy.special import factorial, binom
import jax 
import jax.numpy as jnp
from flax import linen as nn 
import optax

import scipy.sparse as sparse
import scipy.sparse.linalg as sLA

from Comp_Quant_Dynam.operators import n_party_op_sparse, lambda_jump_operators,  build_liouvillian
import Comp_Quant_Dynam.operators as ops 
from scipy.integrate import solve_ivp 

def example_func(x):
    """
    Example function to demonstrate the repository structure.
    Returns the ground state wavefunction of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    return 1 / pi ** (1 / 4) * exp(-x ** 2 / 2)


###################### Solution sheet 2 ######################

def create_xvals(L, npoints, endpoint=True):
    """
    Creates a grid of 'npoints' evenly spaced values between -L/2 and L/2.
    The 'endpoint' parameter determines whether the endpoint L/2 is included in the grid.
    Returns the grid of x values and the grid spacing dx.
    """
    xvals = np.linspace(-L / 2, L / 2, npoints, endpoint=endpoint)
    dx = xvals[1] - xvals[0]
    return xvals, dx


###################### Solution sheet 3 ######################

def FT(psi, x, k):
    """
    Computes the discrete Fourier transform of the wavefunction `psi` defined on the grid `x` to the momentum space grid `k`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return np.sum(psi * exp(-1j * np.outer(k, x)), axis=1)

def iFT(phi, x, k):
    """
    Computes the inverse discrete Fourier transform of the momentum space wavefunction `phi` defined on the grid `k` to the position space grid `x`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return 1 / npoints * np.sum(phi * exp(1j * np.outer(x, k)), axis=1)

def gaussian_wave_packet(x, x0, sigma, p0):
    """
    Creates a Gaussian wave packet centered at position `x0` with width `sigma` and momentum `p0`.
    """
    norm = 1 / np.sqrt(np.sqrt(2 * np.pi) * sigma)
    # p0 * x0 is a global phase factor that we can ignore, so we can omit it
    #  in the expression for the wave packet.
    return norm * exp(-(x - x0) ** 2 / (4 * sigma ** 2) + 1j * p0 * x)

def create_tvecs(tsteps, dt):
    """
    Creates a time vector for steps'' time steps with time step size 'dt'.
    Returns the time vector of length tsteps+1, starting from 0 to tsteps*dt.
    """
    return np.linspace(0, tsteps * dt, tsteps + 1) # will have length tsteps + 1


###################### Exercise sheet 4 ######################
def idx2state(N1, N2, i):
    """
    Converts a single index `i` to a 'state' in the product Hilbert space |n1, n2> of dimension N1 x N2.
    """
    assert i >= 0 and i < N1 * N2, "Index out of bounds"
    n1 = i // N2
    n2 = i % N2
    state = [n1, n2]
    return state 
 
def state2idx(N1, N2, state):
    """
    Converts a `state` |n1, n2> from the product Hilbert space of dimension `N1 x N2` to a single index `i`.
    """
    n1 = state[0]
    n2 = state[1]
    if n1 < 0 or n1 >= N1 or n2 < 0 or n2 >= N2:
        i = -1 # return -1 if the state is out of bounds
    else:
        i = n1 * N2 + n2
    return i
    

###################### Solution sheet 4 ######################
def create_coherent_state(N, alpha):
    """
    Creates a coherent state |alpha> in the Fock basis of dimension `N` with complex amplitude `alpha`.
    The coherent state is defined as:
    |alpha> = exp(-|alpha|^2/2) sum_{n=0}^{N-1} (alpha^n / sqrt(n!)) |n>
    """

    nvec = np.arange(N)
    state = exp(-np.abs(alpha) ** 2 / 2) * np.power(alpha, nvec) / sqrt(factorial(nvec))
    return state

def expectation_value(state, operator):
    """
    Computes the expectation value of an `operator` in a given `state`.
    The `operator` argument can be either a single operator or an iterable of operators.
    If it is an iterable, the function returns a vector of expectation values for each operator.
    """
    n_obsv, operator = _check_if_sized(operator)
    if n_obsv > 1:
        return np.array([expectation_value(state, op) for op in operator])

    return np.vdot(state, operator @ state)

def _check_if_sized(obsv_vec):
    """
    Helper function to check if the input `obsv_vec` is an iterable of operators or a single operator, and to determine the number of observables.
    If `obsv_vec` is a single operator or a Sequence containing a single operator, it returns (1, obsv_vec).
    If `obsv_vec` is an iterable of operators, it returns (n_obsv, obsv_vec) where n_obsv is the number of observables.
    """
    if isinstance(obsv_vec, Iterable) and not isinstance(obsv_vec, (str, bytes)) and getattr(obsv_vec, "ndim", None) != 2:
        if isinstance(obsv_vec, Sequence):
            n_obsv = len(obsv_vec)
        else:
            # e.g. generator: materialize once so length is defined
            obsv_vec = tuple(obsv_vec)
            n_obsv = len(obsv_vec)
        if n_obsv == 1:
            obsv_vec = obsv_vec[0] # if there is only one observable, return it as a single operator instead of a list
    else:
        n_obsv = 1

    return n_obsv, obsv_vec


###################### Exercise sheet 7 ######################

def CSS(N, theta, phi):
    """
    Returns the coefficients of the coherent spin state (CSS) |theta, phi> in the Dicke basis of dimension N+1.
    The CSS is defined as:
    
    |theta, phi> = sum_{k=0}^N sqrt(binomial(N,k)) * (cos(theta/2)^k * sin(theta/2)^(N-k) * exp(i * k * phi)) |k>
    
    where |k> is the Dicke state with k 0-spins (spin up) and N-k 1-spins (spin down).
    Note that the CSS is a superposition of Dicke states with different numbers of excitations, and the coefficients depend on the angles theta and phi.
    The CSS is a generalization of the coherent state for spin systems, and it can be used to describe states that are localized around a specific point on the Bloch sphere.
    """
    
    # exceptions to avoid 0^0
    if theta == np.pi:
        return np.eye(1, N + 1, 0)[0]
    elif theta == 0:
        return np.eye(1, N + 1 , N)[0]
    else:
        kvec = np.arange(0, N + 1)
        trigonometric_part = cos(theta / 2) ** kvec * sin(theta / 2) ** (N - kvec) * exp(1j * kvec * phi)
        return trigonometric_part * sqrt(binom(N, kvec))

def proj_CSS(psi, N, theta, phi):
    """
    Computes the projection of a state `psi` onto a coherent spin state (CSS) defined by angles `theta` and `phi` for a system of `N` spins.
    """
    
    css_state = CSS(N, theta, phi)
    return np.abs(psi.conj().T @ css_state) ** 2

def Husimi_th_ph(N, psi, nth, nph):
    """
    Computes the Husimi distribution of a state `psi` on a grid of angles `theta` and `phi` for a system of `N` spins.
    The grid is defined by `nth` points in the theta direction ([0, pi]) and `nph` points in the phi direction ([0, 2*pi)).
    Returns the grid of theta and phi values and the corresponding Husimi distribution values.
    """
    Theta = np.linspace(0, pi, nth, endpoint=True)
    Phi = np.linspace(0, 2 * pi, nph, endpoint=False)
    # container for Husimi function
    H = np.zeros((nth, nph))
    # calculate H on the grid
    for x_idx, theta in enumerate(Theta):
        for y_idx, phi in enumerate(Phi):
            H[x_idx, y_idx] = proj_CSS(psi, N, theta, phi)
    return Theta, Phi, H


def Husimi_z_phi(N, psi, nz, nph):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `phi` for a system of `N` spins.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `nph` points in the phi direction ([0, 2*pi)).
    Returns the grid of z and phi values and the corresponding Husimi distribution values.
    """

    Z = np.linspace(-1, 1, nz, endpoint=True)
    th = arccos(Z)
    Phi = np.linspace(0, 2 * pi, nph, endpoint=False)
    H = np.zeros((nz, nph))
    for x_idx, theta in enumerate(th):
        for y_idx, phi in enumerate(Phi):
            H[x_idx, y_idx] = proj_CSS(psi, N, theta, phi)
    return Z, Phi, H
    

def Husimi_front(N, psi, nz, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `y` for a system of `N` spins, looking from the +x direction.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of z and y values and the corresponding Husimi distribution values.
    """

    Z = np.linspace(-1, 1, nz, endpoint=True)
    Y = np.linspace(-1, 1, ny, endpoint=True)
    H = np.zeros((nz, ny))
    mask = np.zeros_like(H, dtype=bool) # make pixels outside of the circle white
    for idx_z, z in enumerate(Z):
        for idx_y, y in enumerate(Y):
            r2 = z ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_z, idx_y] = 0
                mask[idx_z, idx_y] = True
            else:
                theta = arccos(z)
                if theta == 0 or theta == np.pi: # in this case phi is undetermined
                    phi = 0
                else:
                    sin_phi = y / sin(theta)
                    if abs(sin_phi) > 1:
                        sin_phi = int(sin_phi / abs(sin_phi)) # set to 1 or -1
                    phi = arcsin(sin_phi) # corresponds to positive x
                H[idx_z, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_z, idx_y] = False
    H = np.ma.array(H, mask=mask)
    return Z, Y, H

def Husimi_back(N, psi, nz, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `y` for a system of `N` spins, looking from the -x direction.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of z and y values and the corresponding Husimi distribution values.
    """
    
    Z = np.linspace(-1, 1, nz, endpoint=True)
    Y = np.linspace(-1, 1, ny, endpoint=True)
    H = np.zeros((nz, ny))
    mask = np.zeros_like(H, dtype=bool)
    for idx_z, z in enumerate(Z):
        for idx_y, y in enumerate(Y):
            r2 = z ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_z, idx_y] = 0
                mask[idx_z, idx_y] = True
            else:
                theta = arccos(z)
                if theta == 0 or theta == pi:
                    phi = 0
                else:
                    sin_phi = y / sin(theta)
                    if abs(sin_phi) > 1:
                        sin_phi = int(sin_phi / abs(sin_phi)) # set to 1 or -1
                    phi = pi - arcsin(sin_phi) # corresponds to negative x
                H[idx_z, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_z, idx_y] = False
    H = np.ma.array(H, mask = mask)
    return Z, Y, H

def Husimi_top(N, psi, nx, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `x` and `y` for a system of `N` spins, looking from the +z direction.
    The grid is defined by `nx` points in the x direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of x and y values and the corresponding Husimi distribution values.
    """
    
    X = np.linspace(-1 , 1, nx, endpoint=True)
    Y = np.linspace(-1 , 1, ny, endpoint=True)
    H = np.zeros((nx, ny))
    mask = np.zeros_like(H, dtype=bool)
    for idx_x, x in enumerate(X):
        for idx_y, y in enumerate(Y):
            r2 = x ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_x, idx_y] = 0
                mask[idx_x, idx_y] = True
            else:
                if r2 > 1: # numerical issues close to the boundary
                    r2 = 1
                z = np.sqrt(1 - r2)
                theta = np.arccos(z)
                # avoid dividing by 0; Gets a bit tricky.
                if np.isclose(x, 0):
                    if y >= 0:
                        phi = pi / 2
                    else:
                        phi = 3 * pi / 2
                elif x > 0:
                    phi = arctan(y / x)
                else:
                    phi = pi + arctan(y / x)
                H[idx_x, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_x, idx_y] = False
    H = np.ma.array(H, mask=mask)
    return X, Y, H


###################### Solution sheet 8 ######################


def partial_trace(psi, M):
    """
    Computes the reduced density matrix obtained by tracing out `M` spins from a pure state `psi` of `N` spins.
    The basis ordering is assumed to be |0...00>, |0...01>, ..., |1...11>, where last spin corresponds to the least significant bit.
    The last (rightmost) `M` spins are traced out, and the resulting reduced density matrix has dimension 2^(N-M) x 2^(N-M).
    """

    N = int(np.log2(len(psi)))
    assert 1 <= M < N, "M must be between 1 and N-1"
    dim_red = 2 ** (N - M)
    dim_trace = 2 ** M
    rho_red = np.zeros((dim_red, dim_red), dtype=complex)
    for i in range(dim_red):
        for j in range(i, dim_red):
            rho_red[i,j] = psi[range(i * dim_trace, (i + 1) * dim_trace)].T @ psi[range(j * dim_trace, (j + 1) * dim_trace)].conj()
    rho_red = rho_red + rho_red.T.conj() - np.diag(np.diag(rho_red)) # make it Hermitian
    return rho_red

def get_evals(rho):
    """
    Computes the eigenvalues of a density matrix `rho` and returns them in descending order.
    Used to compute the entanglement spectrum of a reduced density matrix, which is the set of eigenvalues of the reduced density matrix obtained by tracing out part of a pure state.
    """

    evals = LA.eigvalsh(rho)
    return np.flip(evals) # eigh returns eigenvalues sorted in ascending order, so need to reverse list

def entanglement_entropy(rho):
    """
    Computes the von Neumann entanglement entropy of a density matrix `rho`.
    The von Neumann entropy is defined as:
    S = -Tr(rho log2(rho)) = -sum_i p_i log2(p_i)
    where p_i are the eigenvalues of rho. The function first computes the eigenvalues of rho, then filters out any eigenvalues that are zero (or very close to zero) to avoid issues with the logarithm, and finally computes the entropy using the formula above.
    """

    evals = get_evals(rho)
    
    ps = entanglement_entropy_from_evals(evals) 
    return ps

def entanglement_entropy_from_evals(evals):
    """
    Computes the von Neumann entanglement entropy from a list of eigenvalues `evals` of a density matrix.
    This function is useful if you already have the eigenvalues of the reduced density matrix and want to compute the entanglement entropy without having to reconstruct the density matrix itself.
    The function filters out any eigenvalues that are zero (or very close to zero) to avoid issues with the logarithm, and then computes the entropy using the formula S = -sum_i p_i log2(p_i).
    """
    evals = np.asarray(evals)
    ps = evals[evals > 1e-12]
    return -np.sum(ps * np.log2(ps))

def trace_half_collective(psi):
    """
    Computes the reduced density matrix obtained by tracing out half of the spins from a pure collective spin state `psi` of `N` spins, where `N` is the total number of spins in the system.
    The basis is assumed to be the Dicke basis, where the state |n> corresponds to n excitations (spin up) and N-n non-excitations (spin down).
    """
    
    N = len(psi) - 1
    rho = psi.conj().reshape(1 , N + 1) * psi.reshape(N + 1, 1)
    rho_red = np.zeros((int(N / 2) + 1, int(N / 2) + 1), dtype=complex)
    pvec = np.arange(N / 2 + 1, dtype=int)
    for i in range(len(rho_red)):
        for j in range(len(rho_red)):
            coeff = np.sqrt(binom(N / 2, i) * binom(N / 2, j))
            rho_red[i,j] = coeff * np.sum(rho[i + pvec, j + pvec] * binom(N / 2, pvec) / np.sqrt(binom(N, i + pvec) * binom(N, j + pvec)))
    return rho_red


###################### Solution sheet 9 ######################

def n_party_idx2state(idx, local_dim, N):
    """
    Converts a single index `idx` to a 'state' in the product Hilbert space of dimension `local_dim^N`.
    The basis ordering is assumed to be |-k,-k,...,-k>, |-k,-k,...,-k+1>, ..., |k,k,...,k>, where k = (local_dim - 1) / 2, and the last spin corresponds to the least significant bit. 
    The function returns a state vector of length `N`, where each entry corresponds to the local state of each spin in the product state.
    """
    state = np.zeros((N,), dtype='int32')
    rest = idx
    for i in range(N - 1):
        base = local_dim ** (N - i - 1)
        div = rest // base
        state[i] = div
        rest = rest % base
    state[N - 1] = rest

    
    return np.int64((state - (local_dim - 1) / 2)) # invert #-1 * 



###################### Solution sheet 10 ######################

class Jastrow(nn.Module):
    """
    Short-range Jastrow variational wavefunction.

    The logarithm of a variational wavefunction for ssa one-dimensional spin chain. It includes nearest-neighbour and
    next-nearest-neighbour spin correlations with trainable parameters J1 and J2.

    log psi(s) = sum_i J1 sigma_i sigma_{i+1} + sum_i J2 sigma_i sigma_{i+2}, where sigma_i = +/- 1.
    """

    @nn.compact
    def __call__(self, s):
        """
        Evaluate the logarithm of the Jastrow wavefunction.

        Args:
            s: Spin configuration with entries 0 or 1.
        """

        sigma_z = 2 * s - 1
        J1 = self.param("J1", nn.initializers.zeros, ())
        J2 = self.param("J2", nn.initializers.zeros, ())
        nn_corr = jnp.sum(
            sigma_z * jnp.roll(sigma_z, -1, axis=-1),
            axis=-1,
        )

        nnn_corr = jnp.sum(
            sigma_z * jnp.roll(sigma_z, -2, axis=-1),
            axis=-1,
        )

        return J1 * nn_corr + J2 * nnn_corr
        
             

def MCMC_Sampler_Metropolis_Hastings(model, params, init_state, num_samples, PRNGkey):
    """ 
    Performs Markov Chain Monte Carlo sampling using the Metropolis-Hastings algorithm.

    The sampler starts from an initial spin configuration and proposes new
    configurations by flipping random spins. For each saved sample, a full
    sweep over the spin chain is performed to reduce autocorrelation.
    """

    def MCMC_step(carry, _):
        s, key = carry

        num_spins = s.shape[0]

        def full_sweep_body(carry, _):
             # perform a full sweep over N_spins to generate minimally autocorrelated samples
            s, key = carry

            key, key_idx, key_accept = jax.random.split(key, 3)

            s_flat = s.ravel()

            # Propose a new state by flipping one random spin
            idx = jax.random.randint(
                key_idx,
                shape=(),
                minval=0,
                maxval=num_spins
            )

            flipped_value = 1 - s_flat[idx]
            s_prime_flat = s_flat.at[idx].set(flipped_value)
            s_prime = s_prime_flat.reshape(s.shape)

            # Compute Metropolis-Hastings acceptance probability
            logpsi_prime = model.apply(params, s_prime)
            logpsi_current = model.apply(params, s)

            p_accept = jnp.minimum(
                1.0,
                jnp.exp(
                    2 * jnp.real(logpsi_prime)
                    -
                    2 * jnp.real(logpsi_current)
                )
            )

            # Accept or reject the proposed state
            u = jax.random.uniform(key_accept)
            accept = u < p_accept

            s_next = jnp.where(accept, s_prime, s)

            return (s_next, key), None

        # One full sweep: about one attempted update per spin
        (next_s, next_key), _ = jax.lax.scan(
            full_sweep_body,
            (s, key),
            None,
            length=num_spins
        )

        return (next_s, next_key), next_s

    _, samples = jax.lax.scan(
        MCMC_step,
        (init_state, PRNGkey),
        None,
        length=num_samples
    )

    return samples

def local_energy_TFIM(params, s, model, B):
    """
    Local energy for the 1D transverse-field Ising model:

        H = - sum_i sigma_z_i sigma_z_{i+1} - B sum_i sigma_x_i

    The spin configuration s is represented using 0/1 spins.
    """

    # Convert 0/1 spins to -1/+1 eigenvalues of sigma_z
    z = 2 * s - 1

    # Diagonal Ising interaction term
    E_diag = -jnp.sum(z * jnp.roll(z, -1))

    # Transverse-field term: flip each spin once
    logpsi_s = model.apply(params, s)
    N = s.shape[0]

    def flip_spin(i):
        return s.at[i].set(1.0 - s[i])

    flipped_states = jax.vmap(flip_spin)(jnp.arange(N))

    logpsi_flipped = jax.vmap(
        lambda sf: model.apply(params, sf)
    )(flipped_states)

    # psi(s') / psi(s) = exp(logpsi(s') - logpsi(s))
    ratios = jnp.exp(logpsi_flipped - logpsi_s)

    E_offdiag = -B * jnp.sum(ratios)

    return jnp.real(E_diag + E_offdiag)


def energy_and_gradient(params, samples, model, B):
    """
    Compute the Monte Carlo variational energy and its VMC gradient:

        grad E = 2 Re[ <O* E_loc> - <O*> <E_loc> ]

    where

        O_k(s) = d log psi(s) / d theta_k.
    """

    # Local energy for every sampled spin configuration
    E_loc = jax.vmap(
        lambda s: local_energy_TFIM(params, s, model, B)
    )(samples)

    # Monte Carlo estimate of the variational energy
    E_mean = jnp.mean(E_loc)

    # Centered local energy
    centered_E = jax.lax.stop_gradient(E_loc - E_mean)

    # Define log psi as a function of parameters and spin configuration
    def logpsi_fn(p, s):
        return model.apply(p, s)

    # O_k(s) = d log psi(s) / d theta_k
    O_tree = jax.vmap(
        jax.grad(logpsi_fn),
        in_axes=(None, 0)
    )(params, samples)

    # Apply the VMC gradient formula to every parameter leaf
    def grad_leaf(O):
        shape = (-1,) + (1,) * (O.ndim - 1)
        centered = centered_E.reshape(shape)

        return 2.0 * jnp.real(
            jnp.mean(jnp.conj(O) * centered, axis=0)
        )

    grads = jax.tree_util.tree_map(grad_leaf, O_tree)

    return E_mean, grads 

class NQS_FFNN(nn.Module):
    """
    Feed-forward neural-network ansatz for a Neural Quantum State.

    The model takes a spin configuration and returns one real number,
    corresponding to log psi(s).
    """

    hidden_dims: tuple = (16,)
    actfunc: callable = nn.tanh

    @nn.compact
    def __call__(self, s):
        # Convert 0/1 spins to -1/+1 spins
        x = 2 * s - 1

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=jax.nn.initializers.lecun_normal(),
                bias_init=jax.nn.initializers.zeros,
            )(x)
            x = self.actfunc(x)

        # Final layer outputs one real number: log psi(s)
        x = nn.Dense(
            1,
            kernel_init=jax.nn.initializers.lecun_normal(),
            bias_init=jax.nn.initializers.zeros,
        )(x)

        return x[..., 0]

def run_vmc_training( model, N_spins, B, N_MC, num_iterations, lr, seed):
    """
    Run VMC ground-state search for a given variational model.
    """
    key = jax.random.PRNGKey(seed)
    init_state = jnp.ones((N_spins,))
    params = model.init(key, init_state)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    energies = []
    for it in range(num_iterations):
        key, sample_key = jax.random.split(key)

        samples = MCMC_Sampler_Metropolis_Hastings(
            model,
            params,
            init_state,
            num_samples=N_MC,
            PRNGkey=sample_key,
        )

        # Continue Markov chain from the last sample
        init_state = samples[-1]

        E_var, grads = energy_and_gradient(
            params,
            samples,
            model,
            B,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        energies.append(float(E_var))

        if it % 20 == 0:
            print(f"Iteration {it:4d} | E_var = {float(E_var):.6f}")

    return params, energies


def local_energy_tilted_TFIM(params, s, model, J, B, g):
    """
    Local energy for tilted TFIM using 0/1 spin configurations.

    H = -J sum_i sigma_z_i sigma_z_{i+1}
        -B sum_i sigma_x_i
        -g sum_i sigma_z_i
    """

    # Convert 0/1 spins to -1/+1 sigma_z eigenvalues
    z = 2 * s - 1

    # Diagonal part: -J zz - g z
    E_diag = -J * jnp.sum(z * jnp.roll(z, -1))
    E_diag += -g * jnp.sum(z)

    # Off-diagonal part: -B sx
    logpsi_s = model.apply(params, s)
    N = s.shape[0]

    def flip_spin(i):
        return s.at[i].set(1.0 - s[i])

    flipped_states = jax.vmap(flip_spin)(jnp.arange(N))

    logpsi_flipped = jax.vmap(
        lambda sf: model.apply(params, sf)
    )(flipped_states)

    ratios = jnp.exp(logpsi_flipped - logpsi_s)

    E_offdiag = -B * jnp.sum(ratios)

    return jnp.real(E_diag + E_offdiag)

def energy_and_gradient_tilted(params, samples, model, J, B, g):
    """
    Compute VMC energy and gradient for the tilted TFIM.
    """

    E_loc = jax.vmap(
        lambda s: local_energy_tilted_TFIM(params, s, model, J, B, g)
    )(samples)

    E_mean = jnp.mean(E_loc)

    centered_E = jax.lax.stop_gradient(E_loc - E_mean)

    def logpsi_fn(p, s):
        return model.apply(p, s)

    O_tree = jax.vmap(
        jax.grad(logpsi_fn),
        in_axes=(None, 0)
    )(params, samples)

    def grad_leaf(O):
        shape = (-1,) + (1,) * (O.ndim - 1)
        centered = centered_E.reshape(shape)

        return 2.0 * jnp.real(
            jnp.mean(jnp.conj(O) * centered, axis=0)
        )

    grads = jax.tree_util.tree_map(grad_leaf, O_tree)

    return E_mean, grads

def run_vmc_training_tilted(
    model,
    N_spins,
    J,
    B,
    g,
    N_MC,
    num_iterations,
    lr,
    seed,
):
    """
    Run VMC ground-state search for the tilted TFIM.
    """

    key = jax.random.PRNGKey(seed)

    init_state = jnp.ones((N_spins,))
    params = model.init(key, init_state)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    energies = []

    for it in range(num_iterations):
        key, sample_key = jax.random.split(key)

        samples = MCMC_Sampler_Metropolis_Hastings(
            model=model,
            params=params,
            init_state=init_state,
            num_samples=N_MC,
            PRNGkey=sample_key,
        )

        init_state = samples[-1]

        E_var, grads = energy_and_gradient_tilted(
            params=params,
            samples=samples,
            model=model,
            J=J,
            B=B,
            g=g,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        energies.append(float(E_var))

        if it % 20 == 0:
            print(f"Iteration {it:4d} | E_var = {float(E_var):.6f}")

    return params, energies

###################### Solution sheet 11 ###################### 

def lambda_hamiltonian(Omega_p, Omega_c, Delta_p, Delta_c):
    """
    Hamiltonian for the 3-level lambda system.

    Basis order:
    |g1>, |g2>, |e>

    delta_2 = Delta_p - Delta_c
    """

    delta_2 = Delta_p - Delta_c

    H = np.array([
        [0,             0,          -Omega_p / 2],
        [0,             delta_2,    -Omega_c / 2],
        [-Omega_p / 2,  -Omega_c / 2, Delta_p]
    ], dtype=complex)

    return H 
    
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

# calculate the steady state, return rho in matrix form
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



def steady_state_lambda(Omega_p, Omega_c, Delta_p, Delta_c, gamma_p, gamma_c, gamma_g):
    """
    Calculate the full steady-state density matrix for the lambda system.
    """
    H = lambda_hamiltonian(Omega_p, Omega_c, Delta_p, Delta_c)
    jumps = lambda_jump_operators(gamma_p, gamma_c, gamma_g)
    L = build_liouvillian(H, jumps)
    rho = rho_ss(L)
    return rho


def steady_rho_eg1(Delta_p, Omega_p, Omega_c, Delta_c, gamma_p, gamma_c, gamma_g):
    """
    Calculate steady-state coherence rho_eg1 = <e|rho|g1>.

    Basis order:
    |g1>, |g2>, |e>

    Therefore rho_eg1 = rho[2, 0].
    """
    rho = steady_state_lambda(Omega_p, Omega_c, Delta_p,Delta_c, gamma_p, gamma_c, gamma_g)
    return rho[2, 0]


def scan_rho_eg1(Delta_p_values, Omega_p, Omega_c, Delta_c,gamma_p,gamma_c, gamma_g):
    """
    Scan Delta_p and return rho_eg1 values.
    """
    rho_values = np.array([steady_rho_eg1(Delta_p, Omega_p, Omega_c, Delta_c, gamma_p, gamma_c, gamma_g)
        for Delta_p in Delta_p_values])
    return rho_values

def evolve_master_equation(L_mat, rho0, t_eval):
    """
    Solve the time-dependent master equation:

        d rho_vec / dt = L_mat rho_vec

    Inputs:
    L_mat  : Liouvillian matrix
    rho0   : initial density matrix
    t_eval : array of times

    """
    dim_H = rho0.shape[0]
    rho0_vec = rho0.reshape(dim_H * dim_H)
    def rhs(t, rho_vec):
        return L_mat @ rho_vec

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        rho0_vec,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11
    )

    rho_t = sol.y.T.reshape((len(t_eval), dim_H, dim_H))
    return sol.t, rho_t 

def mcwf_trajectory(H,jumps, psi0, t_eval,seed=None,allow_jumps=True):
    """
    Calculate one Monte Carlo wave function trajectory.

    H      : Hamiltonian
    jumps  : list of jump operators
    psi0   : initial state vector
    t_eval : time array
    """

    rng = np.random.default_rng(seed)
    dim_H = H.shape[0]
    psi = np.array(psi0, dtype=complex)
    psi = psi / np.sqrt(np.vdot(psi, psi).real)
    psis = np.zeros((len(t_eval), dim_H), dtype=complex)
    psis[0] = psi
    jump_times = []
    jump_channels = []

    # effective non-Hermitian Hamiltonian
    CdagC_sum = np.zeros_like(H, dtype=complex)

    for C in jumps:
        CdagC_sum += C.conj().T @ C
    H_eff = H - 0.5j * CdagC_sum

    for n in range(len(t_eval) - 1):
        dt = t_eval[n + 1] - t_eval[n]
        U_eff = expm(-1j * H_eff * dt)
        psi_no_jump = U_eff @ psi
        norm2 = np.vdot(psi_no_jump, psi_no_jump).real
        p_jump = 1.0 - norm2
        p_jump = np.clip(p_jump, 0.0, 1.0)
        
        if allow_jumps and rng.random() < p_jump:
            rates = np.array([
                np.vdot(C @ psi, C @ psi).real
                for C in jumps
            ])

            rate_sum = np.sum(rates)

            if rate_sum > 0:
                probabilities = rates / rate_sum
                jump_index = rng.choice(len(jumps), p=probabilities)
                psi = jumps[jump_index] @ psi
                psi = psi / np.sqrt(np.vdot(psi, psi).real)
                jump_times.append(t_eval[n + 1])
                jump_channels.append(jump_index)

            else:
                psi = psi_no_jump / np.sqrt(norm2)

        else:
            psi = psi_no_jump / np.sqrt(norm2)

        psis[n + 1] = psi

    populations = np.abs(psis)**2
    rho_eg1 = psis[:, 2] * np.conj(psis[:, 0])

    result = {
        "t": t_eval,
        "psi": psis,
        "populations": populations,
        "rho_eg1": rho_eg1,
        "jump_times": np.array(jump_times),
        "jump_channels": np.array(jump_channels)
    }

    return result


def mcwf_average(H, jumps, psi0, t_eval, ntraj, seed=None):
    """
    Average observables over many Monte Carlo wave function trajectories.
    """

    rng = np.random.default_rng(seed)
    dim_H = H.shape[0]
    populations_avg = np.zeros((len(t_eval), dim_H), dtype=float)
    rho_eg1_avg = np.zeros(len(t_eval), dtype=complex)

    trajectories = []
    for n in range(ntraj):
        traj_seed = rng.integers(0, 2**32 - 1)
        traj = mcwf_trajectory(H, jumps, psi0, t_eval, seed=traj_seed, allow_jumps=True)
        populations_avg += traj["populations"]
        rho_eg1_avg += traj["rho_eg1"]
        trajectories.append(traj)

    populations_avg = populations_avg / ntraj
    rho_eg1_avg = rho_eg1_avg / ntraj
    result = { "t": t_eval, "populations": populations_avg,"rho_eg1": rho_eg1_avg, "trajectories": trajectories }

    return result