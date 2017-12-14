'''
   Tests the Cauchy point algorithm
'''
from moola import *
import numpy as np

N = 5
x0 = NumpyPrimalVector(np.zeros(N))
l = NumpyPrimalVector(-np.array(list(range(N)))-1.)
l[-1] = -inf 
u = NumpyPrimalVector(np.array(list(range(N)))+1.)
u[-1] = inf 


def test():

    # Define the function q(x) := x^TGx + d^tx.
    def G(x):
      return NumpyPrimalVector(np.zeros(N))

    # Test lower bounds
    d = NumpyPrimalVector(np.ones(N))
    xc = misc.compute_cauchy_point(G, d, x0, l, u)
    assert (xc.data == np.array([-1., -2., -3., -4., -inf])).all()
    print("Test passed")

    # Test upper bounds
    d = NumpyPrimalVector(-np.ones(N))
    xc = misc.compute_cauchy_point(G, d, x0, l, u)
    assert (xc.data == np.array([1., 2., 3., 4., inf])).all()
    print("Test passed")

    # Test Cauchy point where G = d = 0 
    d = NumpyPrimalVector(np.zeros(N))
    xc = misc.compute_cauchy_point(G, d, x0, l, u)
    assert (xc.data == x0.data).all()
    print("Test passed")

    # Find a Cauchy point which is not at the bounds 
    def G(x):
      Gdiag = np.zeros(N)
      Gdiag[-1] = 8
      return NumpyPrimalVector(Gdiag*x.data)

    d = NumpyPrimalVector(2*np.ones(N))
    xc = misc.compute_cauchy_point(G, d, x0, l, u)
    assert (xc.data - np.array(5*[1/4.]) < 1e-12).all()
    print("Test passed")
