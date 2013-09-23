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
        return NumpyVector(2*arr)

init_control = NumpyVector(random(12))

obj = MyFunctional()
prob = Problem(obj, init_control)


options = {}
options["line_seach"] = None

solver = SteepestDescent(options=options)
print solver
sol = solver.solve(prob)
print sol

assert sol["Optimizer"].norm("L2") < 1e-10
assert sol["Number of iterations"] == 1
