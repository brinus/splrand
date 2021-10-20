''' Main module providing class ProbabilityDensityFunction
'''

#pylint: disable=invalid-name,redefined-outer-name

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import triang
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction:
    '''Class defining the pdf from a set of datas (x,pdf(x))
    '''
    def __init__(self,x,y,spline_order):
        ''' x and y are two numpy arrays sampling the pdf on a grid of values. 
           spline_order is used to define the order of the spline used for    
           calculating the pdf.
        '''
        self.x = x
        self.y = y
        self.spline_order = spline_order
        self.pdf_spline = InterpolatedUnivariateSpline(self.x,self.y,k=self.spline_order)

    def __call__(self,z):
        return self.pdf_spline(z)

    def probability(self,start,stop):
        ''' Calculates the probability in the (start,stop) interval with the
            pdf given by the spline. If start > stop, they are swapped.
        '''
        if start > stop:
            start, stop = stop, start
        return self.pdf_spline.integral(start,stop)

    def sampler(self, n):
        """ This function returns n values distribuited as the pdf_spline.
            The sampling is done calculating the inverse of the cumulative.
            The cdf is taken as the antiderivative of the pdf_spline.
        """
        cdf_spline = self.pdf_spline.antiderivative()
        ppf_spline = InterpolatedUnivariateSpline(cdf_spline(self.x),self.x,k=self.spline_order)

        rnd_values = np.random.random(n)
        sampled_values = ppf_spline(rnd_values)
        return sampled_values

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
    plt.plot(z, pdf_to_sample.pdf_spline(z),label='3rd grade spline')
    plt.errorbar(x,y,fmt='.')
    print(pdf_to_sample.probability(0,0.5))
    w = pdf_to_sample.sampler(100000)
    plt.hist(w,100,density=True)

    plt.legend()
    plt.show()
