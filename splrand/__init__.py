
import numpy as np
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
            pdf given by the spline. It requires start <= stop.
        """
        if start > stop :
            tmp = start
            start = stop
            stop = tmp
        return self.spline.integral(start,stop)
        
    def sampler(self, rnd):
        """ rnd is a single random value in [0,1] or a np array of random values\
            This function returns len(rnd) values distribuited as the pdf_spline.
            The sampling is done calculating the inverse of the cumulative. The \
            cdf is taken as the antiderivative of the pdf_spline.
        """
        cdf_spline = self.pdf_spline.antiderivative()
        ppf_spline = InterpolatedUnivariateSpline(cdf_spline(self.x),self.x,k=self.spline_order)
        return ppf_spline(rnd)
