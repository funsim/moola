'''
  Solves min_x 1/3 * (x[0]**3 + x[1]**3 + ... +x[n-1]**3),
     x in R^n
'''
from moola import *
import numpy as np
from numpy.random import random

class MyFunctional(ObjectiveFunctional):
    def __call__(self, x):
        arr = x.data
        return 1/3.*sum(arr**3)

    def gradient(self, x):
        arr = x.data
        return NumpyVector(arr**2)

init_control = NumpyVector(random(5))
obj = MyFunctional()
prob = Problem(obj)

options = {}
options["gtol"] = 1e-20

solver = SteepestDescent(tol=1e-200, options=options)
print solver
sol = solver.solve(prob, init_control)
print sol

assert sol["Optimizer"].norm("L2") < 1e-9
assert sol["Number of iterations"] < 50
