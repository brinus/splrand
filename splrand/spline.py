
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import triang
from scipy.interpolate import InterpolatedUnivariateSpline


class ProbabilityDensityFunction:
    """Class defining the pdf from a set of datas (x,pdf(x))
    """
    def __init__(self,x,y,spline_order):
       """ x and y are two numpy arrays sampling the pdf on a grid of values. \
           spline_order is used to define the order of the spline used for    \
           calculating the pdf.
       """
       self.x = x
       self.y = y
       self.spline_order = spline_order
       self.pdf_spline = InterpolatedUnivariateSpline(self.x,self.y,k=self.spline_order)
       
    def __call__(self,z):
        return self.pdf_spline(self.z)
        
    def probability(self,start,stop):
        """ Calculates the probability in the (start,stop) interval with the  \
            pdf given by the spline. If start > stop, they are swapped.
        """
        if start > stop:
            tmp = start
            start = stop
            stop = tmp
        return self.pdf_spline.integral(start,stop)
        
    def sampler(self, rnd):
        """ rnd is a single random value in [0,1] or a np array of random values\
            This function returns len(rnd) values distribuited as the pdf_spline.
            The sampling is done calculating the inverse of the cumulative. The \
            cdf is taken as the antiderivative of the pdf_spline.
        """
        cdf_spline = self.pdf_spline.antiderivative()
        ppf_spline = InterpolatedUnivariateSpline(cdf_spline(self.x),self.x,k=self.spline_order)
        return ppf_spline(rnd)
        
        
        
        
def data_orderer(x,y):
    """ Takes two np arrays in input and orders the 1st ascending, the 2nd as  \
        the 1st.
    """
    p = x.argsort()
    x = x[p]
    y = y[p]
    return x,y
    
def triang_pdf(z):
    """ Defining the triangular pdf for testing the various functions. z can be \
        an array or a single value.
    """
    return triang.pdf(z,0.5)
    
    
def sampling_a_pdf(pdf, n, start, stop,):
    """ Takes a function as pdf and samples n points from it in [start,stop]   \
        interval. If start <= stop, the two are swapped. If the pdf is not     \
        normalized (order of tolerance 1e-7), it is divided by its integral.
    """
    if start > stop:
        tmp = start
        start = stop
        stop = tmp
    
    norm = integrate.quad(pdf, start, stop)[0]
    #print(norm)
        
    np.random.seed(283847)
    x = (stop-start) * np.random.random(n) + start
    y = pdf(x)/norm
    return x,y
    
    
    
if __name__ == '__main__':
    n = int(1e5)
    x,y = sampling_a_pdf(triang_pdf,n,0,1.)
    x,y = data_orderer(x,y)
    pdf_to_sample = ProbabilityDensityFunction(x,y,3)
    z=np.linspace(0,1.,int(1e3))
    plt.plot(z, pdf_to_sample.pdf_spline(z),label='3rd grade spline')
    #plt.errorbar(x,y,fmt='.')
    print(pdf_to_sample.probability(0,0.5))
    
    
    plt.legend()
    plt.show()
    
    
        
       
       
        

