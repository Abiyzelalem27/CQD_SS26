

# standard numerics
import math
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial 
<<<<<<< HEAD
import Comp_Quant_Dynam.utility
import Comp_Quant_Dynam.hamiltonians 
import Comp_Quant_Dynam.plotting 


# Comp_Quant_Dynam/__init__.py

from .hamiltonians import (
    sigmax,
    sigmay,
    sigmaz,
    eigenfunction,
    double_well,
)
__all__ = [
    "sigmax",
    "sigmay",
    "sigmaz",
    "eigenfunction",
    "double_well",
    "utility",
    "hamiltonians",
    "plotting", 
]
=======

from .hamiltonians import (sigmax, sigmay, sigmaz, eigenfunction, double_well) 
  
__all__ = ["sigmax", "sigmay", "sigmaz", "eigenfunction", "double_well"]   
>>>>>>> 32c83e9ff15b970e77206e6886ad427876d2a2f4
