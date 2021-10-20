''' Testing main functions from splrand/spline.py
'''

#pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name,no-self-use

import unittest

import numpy as np
import matplotlib.pyplot as plt

from splrand import spline

class TestSpline(unittest.TestCase):
    def test1(self):
        N = int(1e5)
        x, y = spline.sampling_a_pdf(spline.triang_pdf, N, 0, 1.)
        x, y = spline.data_orderer(x,y)
        pdf_to_sample = spline.ProbabilityDensityFunction(x,y,3)
        z = np.linspace(0, 1., int(1e3))
        plt.plot(z, pdf_to_sample.pdf_spline(z),label='3rd grade spline')
        #plt.errorbar(x,y,fmt='.')
        print(pdf_to_sample.probability(0,0.5))

        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main(exit=False)
