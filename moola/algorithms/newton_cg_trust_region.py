from optimisation_algorithm import *
from math import sqrt
from IPython import embed
class TrustRegionNewtonCG(OptimisationAlgorithm):
    ''' 
    An implementation of the trust region NewtonCG method 
    described in Wright 2006, section 7. 
    '''
    def __init__(self, options={}, **args):
        '''
        Initialises the trust region Newton CG method. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Not supported yet - must be None. 
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
          '''
        print "Test"

        # Set the default options values
        self.tol = options.get("tol", None)
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", 2)

        if self.tol is not None:
            print 'tol parameter not yet supported. Will be ignored.'

        # method specific options:
        self.tr_Dmax = options.get("ncg_Dmax", 1) # overall bound on the step lengths
        self.tr_D0 = options.get("ncg_D0", 0.5*self.tr_Dmax) # current bound on the step length
        self.eta = options.get("eta", 1./8)

        assert 0 <= self.eta < 1./4
        assert self.tr_Dmax > 0
        assert 0 < self.tr_D0 < self.tr_Dmax
    
    def __str__(self):
        s = "Trust region Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s


    def mk(self, obj, xk, pk):
        ''' Evaluates the quadratic approximation '''
        if pk == 0:
            return obj(xk)
        else:
            return obj(xk) + obj.derivative(xk).apply(pk) + 0.5 * obj.hessian(xk)(pk).apply(pk)

    def get_tau(self, obj, z, d, delta):
        """ Function to find tau such that p = pj + tau.dj, and ||p|| = delta. """

        dot_z_d = z.inner(d)
        len_d_sqrd = d.inner(d)
        t = sqrt(dot_z_d**2 - len_d_sqrd * (z.inner(z) - delta**2)) / len_d_sqrd

        taup = - dot_z_d + t
        taum = - dot_z_d - t
        return taup, taum

    def compute_pk_cg_steihaug(self, obj, x, D, eps=1e-10):
        ''' Solves min_pk fk + grad fk(p) + 0.5*p^T H p using the CG Steighaug method '''  
        z = x.copy()
        z.zero()
        r = obj.gradient(x)
        d = -r

        if r.norm() < eps:
            return z

        cg_iter = 0
        while cg_iter < 20:
            cg_iter += 1
            Hd = obj.hessian(x)(d)
            curv = Hd.apply(d)

            if curv <= 0:
                tau = self.get_tau(obj, z, d, D)[0]
                print "curv <= 0.0, hence tau = " + repr(tau)
                return x + tau*d

            rtr = r.inner(r)
            alpha = rtr / curv
            z_old = z.copy()
            z.axpy(alpha, d)

            if z.norm() >= D:
                tau = self.get_tau(obj, z_old, d, D)[0]
                print "|z| >= Delta, hence tau = " + repr(tau)
                return x + tau*d

            r.axpy(alpha, Hd.primal())

            if r.norm() < eps:
                print "|r| < length_test"
                return z

            rtr, rtr_old = r.inner(r), rtr
            beta = rtr / rtr_old

            d = -r + beta*d


    def solve(self, problem, xk):
        ''' Solves the optimisation problem with the trust region Newton-CG method. 
            Arguments:
             * problem: The optimisation problem.
             * xk: The initial guess

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        print self
            
        print "Doing another iteration"
        obj = problem.obj
        opt_iter = 0
        Dk = self.tr_D0

        while True:
            res = obj.derivative(xk)
            if self.check_convergence(opt_iter, None, None, res):
                break

            self.display(opt_iter, None, None, res)

            # compute trust region guess
            pk = self.compute_pk_cg_steihaug(obj, xk, Dk)

            rhok = (obj(xk) - obj(xk + pk)) / (self.mk(obj, xk, 0) - self.mk(obj, xk, pk))

            if rhok < 1./4:
                Dk *= 1./4

            elif rhok > 3./4 and pk.norm() == Dk:
                Dk = min(2*Dk, self.tr_Dmax)

            if rhok > self.eta:
                xk += pk

            opt_iter += 1
            

        self.display(opt_iter, None, None, res)
        sol = {'Optimizer': xk,
               'Number of iterations': opt_iter,
               'Functional value at optimizer': None }
        return sol
                
 
