''' Testing main functions from splrand/spline.py
'''

#pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name,no-self-use

import unittest

import numpy as np
import matplotlib.pyplot as plt

from splrand import spline

class TestSpline(unittest.TestCase):
    def test_data_order(self):
        x_test = np.array([0.,5.25,-1.,2.,4.,3])
        y_test = np.array([1,5,0,2,4,3])
        x = np.array([-1.,0.,2.,3,4.,5.25])
        y = np.array([0,1,2,3,4,5])
        x_test, y_test = spline.data_orderer(x_test,y_test)
        np.testing.assert_array_almost_equal(x_test, x)
        np.testing.assert_array_almost_equal(y_test, y)
        



    def test1(self):
        N = int(1e3)
        x, y = spline.sampling_a_pdf(spline.triang_pdf, N, 0, 1.)
        x, y = spline.data_orderer(x,y)
        pdf_to_sample = spline.ProbabilityDensityFunction(x,y,3)
        z = np.linspace(0, 1., int(1e3))
        plt.plot(z, pdf_to_sample.pdf_spline(z),label='3rd grade spline')
        #plt.errorbar(x,y,fmt='.')
        print(pdf_to_sample.probability(0,0.5))

        #plt.legend()
        #plt.show()

if __name__ == '__main__':
    unittest.main(exit=False)
