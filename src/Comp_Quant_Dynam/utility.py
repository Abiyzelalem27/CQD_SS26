import numpy as np   # standard numerics library
from numpy import linalg as LA
from collections.abc import Iterable, Sequence
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan, sqrt, exp
from scipy.special import factorial, binom
import jax
import jax.numpy as jnp
from flax import linen as nn 
import optax


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

def run_vmc_training(
    model,
    N_spins,
    B,
    N_MC,
    num_iterations,
    lr,
    seed,
):
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
            model=model,
            params=params,
            init_state=init_state,
            num_samples=N_MC,
            PRNGkey=sample_key,
        )

        # Continue Markov chain from the last sample
        init_state = samples[-1]

        E_var, grads = energy_and_gradient(
            params=params,
            samples=samples,
            model=model,
            B=B,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        energies.append(float(E_var))

        if it % 20 == 0:
            print(f"Iteration {it:4d} | E_var = {float(E_var):.6f}")

    return params, energies