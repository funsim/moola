'''
  Solves min_x 1/2 * (x[0]**2 + x[1]**2 + ... +x[n-1]**2),
     x in R^n
'''
from moola import *
import numpy as np
from numpy.random import random

class MyFunctional(ObjectiveFunctional):
    def __call__(self, x):
        arr = x.data
        return 0.5*sum(arr**2)

    def gradient(self, x):
        arr = x.data
        return NumpyVector(arr)

init_control = NumpyVector(random(5))

obj = MyFunctional()
prob = Problem(obj)

# Solve with steepest descent
options = {}
solver = SteepestDescent(options=options)
sol = solver.solve(prob, init_control)
assert sol["Optimizer"].norm("L2") < 1e-10
assert sol["Number of iterations"] == 2

# Solve with Fletcher-Reeves method 
options = {'gtol': 1e-10}
solver = FletcherReeves(options=options)
sol = solver.solve(prob, init_control)
assert sol["Optimizer"].norm("L2") < 1e-10
