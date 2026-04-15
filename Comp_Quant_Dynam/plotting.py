import numpy as np
import matplotlib.pyplot as plt


#################### Exercise sheet 1 ####################


def plot_func(func, k):
    """
    Plots the function 'func' for a given value of 'k'.
    Useful for creating interactive plots in Jupyter notebooks.
    """

    plt.figure(2)
    plt.clf()
    x = np.linspace(0, 2*np.pi, num=1000)
    plt.plot(x,func(x, k))
    plt.xlabel('x')
    plt.ylabel('func(x, k)')

    return plt.gcf()


#################### Solution sheet 1 ####################


def plot_eigenstate(n, x, evals, evecs):
    """
    Plots the 'n'-th eigenstate of an operator in the position-space representation for a given grid 'x', eigenvalues 'evals', and eigenvectors 'evecs'.
    """
    
    dx = x[1] - x[0]
    evec_amp_n = np.abs(evecs[:,n]) ** 2 / dx
    sig_digits = 5 # number of significant digits to display in the title
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, evec_amp_n)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.6)
    # add labels and legends
    ax.set_title("$E_n=$"+f'{evals[n]:.{sig_digits}f}')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\\phi_n(x)|^2$")
    
    return fig