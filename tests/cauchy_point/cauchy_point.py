'''
   Tests the Cauchy point algorithm
'''
from moola import *
import numpy as np

N = 5
x0 = NumpyVector(np.zeros(N))
l = NumpyVector(-np.array(range(N))-1)
u = NumpyVector(np.array(range(N))+1)

print "Lower bound", l.data
print "Upper bound", u.data

# Define the function q(x) := x^TGx + d^tx.
d = NumpyVector(np.ones(N))
def G(x):
  return NumpyVector(np.zeros(N))

xc = misc.compute_cauchy_point(G, d, x0, l, u)
assert (xc.data - np.array([-1., -2., -3., -4., -5.]) < 1e-12).all()
print "Test passed"
