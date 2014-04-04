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

        return NumpyDualVector(dr)

    def hessian(self, val, vec):
        x, y = val.data
        dxx = 2. + 1200.*x**2 - 400.*y
        dxy = -400.*x
        dyy = 200.
        d2v = (dxx * vec[0] + dxy * vec[1], dxy*vec[0] +dyy*vec[1] )
        return NumpyDualVector(d2v)
                

obj = MyFunctional()

x_init = NumpyPrimalVector((-3, -4))
prob = Problem(obj)

x_opt = (1, 1)
f_opt = 0
options = {'disp':2, 'tol': None, 'gtol': 1e-16, 'maxiter': 200, 'mem_lim': 2}

# Solve the problem with the steepest descent method
#solver = SteepestDescent(tol=None, options=options)
#sol = solver.solve(prob, x_init.copy())
#assert max(abs(sol["Optimizer"].data + x_opt)) < f_opt + 1e-9
#assert sol["Number of iterations"] < 50
#print sol['Optimizer'].data
#print '\n\n\n'

# Solve the problem with the Fletcher-Reeves method
solver = NonLinearCG(options=options)
sol = solver.solve(prob, x_init.copy())
#assert max(abs(sol["Optimizer"].data + x_opt)) < f_opt + 1e-9
#assert sol["Number of iterations"] < 50
print sol['Optimizer'].data
#print '\n\n\n'

# Solve the problem with the BFGS method
#solver = BFGS(options=options)
#sol = solver.solve(prob, x_init.copy())
#print sol['Optimizer'].data
#print '\n\n\n'
