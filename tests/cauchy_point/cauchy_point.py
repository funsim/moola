'''
   Tests the Cauchy point algorithm
'''
from moola import *
import numpy as np
inf = float("inf")

N = 5
x0 = NumpyVector(np.zeros(N))
l = NumpyVector(-np.array(range(N))-1.)
l[-1] = -inf 
u = NumpyVector(np.array(range(N))+1.)
u[-1] = inf 

print "Lower bound", l.data
print "Upper bound", u.data

# Define the function q(x) := x^TGx + d^tx.
def G(x):
  return NumpyVector(np.zeros(N))

d = NumpyVector(np.ones(N))
xc = misc.compute_cauchy_point(G, d, x0, l, u)
assert (xc.data == np.array([-1., -2., -3., -4., -inf])).all()
print "Test passed"

d = NumpyVector(-np.ones(N))
xc = misc.compute_cauchy_point(G, d, x0, l, u)
assert (xc.data == np.array([1., 2., 3., 4., inf])).all()
print "Test passed"
