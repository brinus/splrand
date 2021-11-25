''' Testing spline_child module
'''

import numpy as np
#import matplotlib.pyplot as plt

from splrand.spline_child import ProbabilityDensityFunction as PDF

x = np.linspace(1, 10, 10)
y = np.random.normal(0, 1, 10)

pdf = PDF(x,y)

pdf.x[4], pdf.y[4] = 1,2

for i, val in enumerate(pdf):
    print(f'pdf[{i}] = {val}')

# plt.errorbar(pdf.x, pdf.y, fmt='r.')
# plt.show()
