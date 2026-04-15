import numpy as np
import numpy.linalg as LA
import Comp_Quant_Dynam.hamiltonians as ham


class Test_HO_eigenstates_exact:

    L = 10
    npoints = 101
    xvals = np.linspace(-L / 2, L / 2, npoints)
    
    def test_HO_ground(self):
        n0 = 0
        expected =  1 / np.pi ** (1 / 4) * np.exp(-self.xvals ** 2 / 2)
        result = ham.HO_eigenstates_exact(n0, self.xvals)
        assert np.allclose(expected, result)

class Test_HO_eigenenergies:
    
    L = 15
    npoints = 2001
    xvals = np.linspace(-L / 2, L / 2, npoints)

    def test_HO_ED(self):
        
        H_pot = ham.HO_potential(self.xvals)
        H_kin = ham.H_kinetic(self.xvals)

        H_mat = H_pot + H_kin

        evals_num, evecs_num = LA.eigh(H_mat)
        evals_exact = ham.HO_eigenenergies_exact(np.arange(evals_num.size))
        print(evals_num[:10])
        print(evals_exact[:10])
        assert np.allclose(evals_num[:10], evals_exact[:10], atol=1e-3)



        