

[![CI](https://github.com/Abiyzelalem27/CQD_SS26/actions/workflows/python_CI.yml/badge.svg)](https://github.com/Abiyzelalem27/CQD_SS26/actions/workflows/python_CI.yml)

[![codecov](https://codecov.io/github/Abiyzelalem27/CQD_SS26/graph/badge.svg)](https://codecov.io/github/Abiyzelalem27/CQD_SS26)



# 📌 Overview

This repository contains the weekly programming exercises for the course “Computational Quantum Dynamics” (Summer Semester 2026).

- Quantum mechanics  
- Linear algebra  
- Numerical analysis  
- Many-body physics  
  
## 🎯 The goal is to build computational tools to:

- Solve the Schrödinger equation  
- Compute eigenstates of quantum Hamiltonians  
- Simulate time evolution  
- Study many-body quantum systems  
- Explore semiclassical and classical limits  


# 📚  Overview

## 1 — Quantum Mechanics Basics and Numerical Discretization

Quantum states live in a Hilbert space, and physical observables are represented by operators.

- Postulates of quantum mechanics (states, observables, measurement, dynamics)  
- Schrödinger equation (time-dependent and time-independent)  
- Wavefunctions in L² space  
- Position and momentum operators  
- Harmonic oscillator as analytical example  

## 2 — Numerical Linear Algebra for Quantum Systems

Quantum mechanics reduces to eigenvalue problems and matrix exponentials.

- Matrix classes: Hermitian, unitary, normal matrices  
- QR algorithm for eigenvalue problems  
- Hessenberg and tridiagonal reduction (Householder transformations)  
- Givens rotations for sparse elimination  
- Sparse matrix formats (COO, CSR, DIA)  
- Krylov subspaces and matrix power methods  
- Lanczos algorithm for sparse Hamiltonians  


## 3 — Split-Step Fourier Method

Quantum time evolution can be split into kinetic and potential parts.

- Trotter and Strang decomposition  
- Position vs momentum space representation  
- Fourier transform of wavefunctions  
- FFT algorithm (O(N log N))  
- Cooley–Tukey method  

## 4 — Many-Body Quantum Systems and Approximations

Many-body quantum systems suffer from exponential scaling (“curse of dimensionality”).

- Harmonic and anharmonic oscillator systems  
- Symmetry reduction techniques  
- Variational principle for ground states  
- Mean-field approximation   
- Gross–Pitaevskii equation  



## 5 — Spin-½ Systems and Collective Quantum Physics

Spin systems provide a minimal model for quantum many-body physics.

- Spin-½ Hilbert space and qubits  
- Bloch sphere representation  
- Pauli matrices and SU(2) algebra  
- Collective spin operators  
- Permutation symmetry and Dicke states  
- Lipkin–Meshkov–Glick (LMG) model  
- Ferromagnetic interactions and transverse field

## 10 — Ground State Search with Variational Monte Carlo

- Variational Monte Carlo approximates ground-state energies without storing the full many-body wavefunction.
- Trainable trial wavefunctions such as the Jastrow ansatz and Neural Quantum States are used.
- Monte Carlo sampling and local energy estimates are used to minimize the variational energy.
- The method is tested on the transverse-field Ising model and the tilted TFIM, with exact diagonalization used as a small-system benchmark.




## 📚 References

* **CQD_SS26 Main Repository (Primary Course Source)**
  Course code repository for Computational Quantum Dynamics (SS26) taught by **Prof. Gärttner** .
  https://github.com/NiklasEuler/CQD_SS26
* **Quantum Information and Quantum Simulation (QIQS) Group in Friedrich Schiller University Jena, Germany**
  Research group and academic environment associated with the course.
  https://qiqs-jena.de/

* **Discrete Fourier Transform (DFT) — Python Numerical Methods (Berkeley)**
  https://pythonnumericalmethods.studentorg.berkeley.edu

* **NumPy Documentation**
  https://numpy.org/doc/stable/index.html