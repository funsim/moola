'''
  Minimises the Rosenbrock function

  f(x, y) = (1 - x^2) + 100*(y - x^2)^2
     
'''
from moola import *

class MyFunctional(Functional):
    def __call__(self, val):
        x, y = val.data
        r = (1 - x)**2 + 100*(y - x**2)**2
        return r

    def derivative(self, val):
        x, y = val.data
        print "current x =", x, y 

        dx = - 2*(1 - x) + 100*2*(y - x**2) * (-2*x)
        dy = 100 * 2 * (y - x**2)
        dr = (dx, dy)

        return NumpyLinearFunctional(dr)

obj = MyFunctional()

x_init = NumpyVector((-3., -4.))
prob = Problem(obj)

x_opt = (1, 1)
f_opt = 0
options = {'gtol': None, 'maxiter': 1000}

# Solve the problem with the steepest descent method
#solver = SteepestDescent(tol=None, options=options)
#sol = solver.solve(prob, x_init)
#assert max(abs(sol["Optimizer"].data + x_opt)) < f_opt + 1e-9
#assert sol["Number of iterations"] < 50

# Solve the problem with the Fletcher-Reeves method
#solver = FletcherReeves(options=options)
#sol = solver.solve(prob, x_init)
#assert max(abs(sol["Optimizer"].data + x_opt)) < f_opt + 1e-9
#assert sol["Number of iterations"] < 50

# Solve the problem with the BFGS method
solver = BFGS(options=options)
sol = solver.solve(prob, x_init)
