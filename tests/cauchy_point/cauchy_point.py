'''
   Tests the Cauchy point calculation
'''
from moola import *
import numpy as np

N = 5
x0 = NumpyVector(np.zeros(N))
l = NumpyVector(-np.ones(N))
u = NumpyVector(np.zeros(N))

# Define the function q(x) := x^TGx + d^tx.
d = NumpyVector(np.ones(N))
def G(x):
  return NumpyVector(np.zeros(N))

xc = misc.compute_cauchy_point(G, d, x0, l, u)

print xc
