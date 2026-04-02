import numpy as np
import Comp_Quant_Dynam.utility as util


class Test_example_function:

    def test_example_func_zero(self):
        x = 0
        expected =  1 / np.pi ** (1 / 4)
        result = util.example_func(x)
        assert np.allclose(expected, result)

    def test_example_func_symmetry(self):
        x = np.array([-1, 1])
        result = util.example_func(x)
        assert np.allclose(result[0], result[1])
