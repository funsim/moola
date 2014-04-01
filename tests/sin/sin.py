'''
  Solves \sum_x_i sin(x_i)
     x in R^n
'''
from moola import *
import numpy as np
from numpy.random import random

class MyFunctional(Functional):
    def __call__(self, x):
        arr = x.data
        return sum(np.sin(arr))

    def gradient(self, x):
        arr = x.data
        return NumpyVector(np.cos(arr))

init_control = NumpyVector(np.ones(1))
obj = MyFunctional()
prob = Problem(obj)

# Solve the problem with the steepest descent method
options = {'gtol': 1e-16}

solver = SteepestDescent(tol=1e-200, options=options)
sol = solver.solve(prob, init_control)
assert max(abs(sol["Optimizer"].data + 1./2*np.pi)) < 1e-9
assert sol["Number of iterations"] < 50

# Solve the problem with the Fletcher-Reeves method
solver = FletcherReeves(options=options)
sol = solver.solve(prob, init_control)
assert max(abs(sol["Optimizer"].data + 1./2*np.pi)) < 1e-9
assert sol["Number of iterations"] < 50

# Solve the problem with the BFGS method
#solver = BFGS(options=options)
#sol = solver.solve(prob, init_control)
