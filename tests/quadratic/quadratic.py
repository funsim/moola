'''
  Solves min_x 1/2 * (x[0]**2 + x[1]**2 + ... +x[n-1]**2),
     x in R^n
'''
from moola import *
import numpy as np

class MyFunctional(ObjectiveFunctional):
    def __eval__(self, x):
        return 0.5*sum(x**2)
    def gradient(self, x):
        return 2*x

init_control = NumpyVector(np.ones(12))

obj = MyFunctional()
prob = Problem(obj, init_control)

solver = SteepestDescent()
sol = solver.solve(prob)

print sol
