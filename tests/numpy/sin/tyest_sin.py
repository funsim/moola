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

    def derivative(self, x):
        arr = x.data
        return NumpyDualVector(np.cos(arr))


def test():
    init_control = NumpyPrimalVector(np.ones(1))
    obj = MyFunctional()
    prob = Problem(obj)


    # Solve the problem with the steepest descent method
    options = {'jtol': 0, 'gtol': 1e-16}
    solver = SteepestDescent(prob, init_control.copy(), options=options)
    sol = solver.solve()
    assert max(abs(sol["control"].data + 1./2*np.pi)) < 1e-9
    assert sol["iteration"] < 10

    print '\n'

    # Solve the problem with the steepest descent from the NonlinearCG method
    options = {'jtol': 0, 'gtol': 1e-16, "beta_rule": "steepest_descent"}
    solver = NonLinearCG(prob, init_control.copy(), options=options)
    sol = solver.solve()
    assert max(abs(sol["control"].data + 1./2*np.pi)) < 1e-9
    assert sol["iteration"] < 10

    print '\n'

    # Solve the problem with the Fletcher-Reeves method
    options = {'jtol': 0, 'gtol': 1e-16, "beta_rule": "fletcher-reeves"}
    solver = NonLinearCG(prob, init_control.copy(), options=options)
    sol = solver.solve()
    assert max(abs(sol["control"].data + 1./2*np.pi)) < 1e-9
    assert sol["iteration"] < 30

    # Solve the problem with the Hager-Zhang method
    options = {'jtol': 0, 'gtol': 1e-16, "beta_rule": "hager-zhang"}
    solver = NonLinearCG(prob, init_control.copy(), options=options)
    sol = solver.solve()
    assert max(abs(sol["control"].data + 1./2*np.pi)) < 1e-9
    assert sol["iteration"] < 10

    print '\n'

    # Solve the problem with the BFGS method
    options = {'jtol': 0, 'gtol': 1e-16, 'mem_lim':0}
    solver = BFGS(prob, init_control.copy(), options=options)
    sol = solver.solve()
    assert max(abs(sol["control"].data + 1./2*np.pi)) < 1e-9
    assert sol["iteration"] < 10
