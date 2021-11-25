''' Main module providing class ProbabilityDensityFunction
'''

#pylint: disable=invalid-name,redefined-outer-name,missing-function-docstring

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    ''' Class defining the pdf from a set of datas (x,pdf(x))
    '''
    def __init__(self, x, y, k=3):
        ''' x and y are two numpy arrays sampling the pdf on a grid of values.
            spline_order is used to define the order of the spline used for
            calculating the pdf.
        '''
        super().__init__(x, y, k=k)

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    @property
    def k(self):
        return self._data[5]

    def __getitem__(self, index):
        return [self._data[0][index], self._data[1][index]]

    def __setitem__(self, *args, **kwargs):
        # raise NotAssignable('PDF elements cannot be modified')
        pass
    def probability(self,start,stop):
        ''' Calculates the probability in the (start,stop) interval with the
            pdf given by the spline. If start > stop, they are swapped.
        '''
        if start > stop:
            start, stop = stop, start
        return self.integral(start,stop)

    def sampler(self, n):
        ''' This function returns n values distribuited as the pdf_spline.
            The sampling is done calculating the inverse of the cumulative.
            The cdf is taken as the antiderivative of the pdf_spline.
        '''
        cdf_spline = self.antiderivative()
        ppf_spline = InterpolatedUnivariateSpline(cdf_spline(self.x),
                                                  self.x, k=self.k)

        rnd_values = np.random.random(n)
        sampled_values = ppf_spline(rnd_values)
        return sampled_values

# class NotAssignable(Exception):
#     def __init__(self, message):
#         pass
