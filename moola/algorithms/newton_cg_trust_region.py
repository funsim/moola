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
        self.tr_Dmax = options.get("ncg_Dmax", 1e5) # overall bound on the step lengths
        self.tr_D0 = options.get("ncg_D0", 1) # current bound on the step length
        self.eta = options.get("eta", 1./8)

        assert 0 <= self.eta < 1./4
        assert self.tr_Dmax > 0
        assert 0 < self.tr_D0 < self.tr_Dmax
    
    def __str__(self):
        s = "Trust region Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s


    def get_tau(self, obj, z, d, D, verify=True):
        """ Function to find tau such that p = pj + tau.dj, and ||p|| = D. """

        
        dot_z_d = z.inner(d)
        len_d_sqrd = d.inner(d)
        len_z_sqrd = z.inner(z)

        t = sqrt(dot_z_d**2 - len_d_sqrd * (len_z_sqrd - D**2))

        taup = (- dot_z_d + t) / len_d_sqrd
        taum = (- dot_z_d - t) / len_d_sqrd

        if verify:
            eps = 1e-8
            if abs((z+taup*d).norm()-D)/D > eps or abs((z+taum*d).norm()-D)/D > eps:
                raise ArithmeticError, "Tau could not be computed accurately due to numerical errors."

        return taup, taum

    def compute_pk_cg_steihaug(self, obj, x, D):
        ''' Solves min_pk fk + grad fk(p) + 0.5*p^T H p using the CG Steighaug method '''  
        z = x.copy()
        z.zero()
        r = obj.gradient(x)
        d = -r
        rtr = r.inner(r)
        rnorm = rtr**0.5
        eps = min(0.5, sqrt(rnorm))*rnorm  # Stopping criteria for CG

        if rnorm < eps:
            print "CG solver converged"
            return z

        cg_iter = 0
        while True:
            print "CG iteration %s" % cg_iter
            cg_iter += 1

            Hd = obj.hessian(x)(d)
            curv = Hd.apply(d)

            # Curvatur test 
            if curv <= 0:
                print "curv <= 0.0"
                taup, taum = self.get_tau(obj, z, d, D)
                pp = z + taup*d
                pm = z + taum*d

                if self.mk(obj, xk, pp) <= self.mk(obj, xk, pm):
                    return pp, True
                else:
                    return pm, True

            alpha = rtr / curv
            z_old = z.copy()
            z.axpy(alpha, d)
            znorm = z.norm()
            print "|z_%i| = %f" % (cg_iter, znorm)

            # Trust region boundary test
            if znorm >= D:
                print "|z| >= Delta"
                tau = self.get_tau(obj, z_old, d, D)[0]
                assert tau >= 0
                return z_old + tau*d, True

            r.axpy(alpha, Hd.primal())
            rtr, rtr_old = r.inner(r), rtr
            rnorm = rtr**0.5

            # CG convergence test
            if rnorm < eps:
                print "CG solver converged"
                return z, False

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

            # Compute trust region point
            pk, is_cauchy_point = self.compute_pk_cg_steihaug(obj, xk, Dk)

            # Evaluate trust region performance
            objxk = obj(xk)
            mkxkpk = objxk + obj.derivative(xk).apply(pk) + 0.5 * obj.hessian(xk)(pk).apply(pk)
            objxkpk = obj(xk + pk)

            rhok = (objxk - objxkpk) / (objxk - mkxkpk)

            if rhok < 1./4:
                Dk *= 1./4
                print "Decreasing trust region radius to %f." % Dk

            elif rhok > 3./4 and is_cauchy_point:
                Dk = min(2*Dk, self.tr_Dmax)
                print "Increasing trust region radius to %f." % Dk

            if rhok > self.eta:
                xk.axpy(1., pk)
                print "Trust region step accepted."
            else:
                print "Rejecting step. Reason: trust region step did not reduce objective."

            opt_iter += 1
            

        self.display(opt_iter, None, None, res)
        sol = {'Optimizer': xk,
               'Number of iterations': opt_iter,
               'Functional value at optimizer': None }
        return sol
