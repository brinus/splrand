''' Testing child class structure for ProbabilityDensityFunction
'''

#pylint: disable=redefined-outer-name,invalid-name

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import triang
from scipy import integrate

from splrand.spline_child import ProbabilityDensityFunction

def data_orderer(x,y):
    ''' Takes two np arrays in input and orders the 1st ascending, the 2nd as
        the 1st.
    '''
    p = x.argsort()
    x = x[p]
    y = y[p]
    return x,y

def triang_pdf(z):
    ''' Defining the triangular pdf for testing the various functions. z can be
        an array or a single value.
    '''
    return triang.pdf(z,0.5)

def sampling_a_pdf(pdf, n, start, stop,):
    ''' Takes a function as pdf and samples n points from it in [start,stop]
        interval. If start <= stop, the two are swapped. If the pdf is not
        normalized (order of tolerance 1e-7), it is divided by its integral.
    '''
    if start > stop:
        start, stop = stop, start

    norm = integrate.quad(pdf, start, stop)[0]
    #print(norm)

    np.random.seed(283847)
    x = (stop-start) * np.random.random(n) + start
    y = pdf(x)/norm
    return x,y

if __name__ == '__main__':
    N = int(1e5)
    x,y = sampling_a_pdf(triang_pdf, N, 0, 1.)
    x,y = data_orderer(x,y)
    pdf_to_sample = ProbabilityDensityFunction(x,y,3)
    print(pdf_to_sample(0.))
    z=np.linspace(0,1.,int(1e3))
    plt.plot(z, pdf_to_sample(z),label='3rd grade spline')
    plt.errorbar(x,y,fmt='.')
    print(pdf_to_sample.probability(0,0.5))
    w = pdf_to_sample.sampler(100000)
    plt.hist(w,100,density=True)

    plt.legend()
    plt.show()
