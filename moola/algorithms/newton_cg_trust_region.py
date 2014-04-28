from optimisation_algorithm import *
class NewtonCGTrustRegion(OptimisationAlgorithm):
    ''' 
    An implementation of the trust region Newton-CG algorithm
    described in Wright 2006, chapter 7. 
    '''
    def __init__(self, tol=None, options={}, **args):
        '''
        Initialises the trust region Newton CG method. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
          '''

        # Set the default options values
        if tol is not None:
            print 'tol argument not supported. Will be ignored.'
            self.tol = tol
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", 2)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)

        # method specific options:
        self.ncg_reltol  = options.get("ncg_reltol", .5)
        self.ncg_maxiter = options.get("ncg_maxiter", 200)
        self.ncg_hesstol = options.get("ncg_hesstol", "default")
    
    def __str__(self):
        s = "Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def solve(self, problem, xk):
        ''' Solves the optimisation problem with the Fletcher-Reeves method. 
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        print self
        if self.ncg_hesstol == "default":
            import numpy
            eps = numpy.finfo(numpy.float64).eps
            self.ncg_hesstol = eps*numpy.sqrt(len(xk))
            
        objective = problem.obj
        opt_iter = 0
        while True:
            if opt_iter >= self.maxiter:
                break
            # compute derivative at current control
            dJ_xk = objective.derivative(xk)
            '''
            The newton step is
            
                d^2 J (xk) dx = -dJ(xk).      [N]
            
            We solve the subproblem [N] approximately.
            '''

            res  = -dJ_xk.copy()       #  Initial residual
            z    = res.primal()        #  Krylov solver iterate
            p    = z.copy()            #    ---------------- 
            dx   = 0 * z               #  Current CG solution of [N]
            err  = res.apply(z)        #  Norm of the residual
              
            Hk = objective.hessian(xk)
            

            
            if self.check_convergence(opt_iter,None,None, res):
                break
            self.display(opt_iter, None, None, res)

            # CG iterations
            cg_tol = min(self.ncg_reltol**2, err**.5) * err   # Stopping criterion for the CG solve
            cg_iter = 0
            cg_break = 0
            while True:
                if cg_iter >= self.ncg_maxiter:
                    cg_break = 1
                # Compute curvature at the current CG iterate.
                Hk_p = Hk(p)
                curve = Hk_p.apply(p)
                print 'cg_iter = {}\tcurve = {}\thesstol = {}'.format(cg_iter, curve,self.ncg_hesstol)
                if curve < 0 and cg_iters == 0:
                    # Fall back to steepest descent.
                    dx = z
                    break
                if 0 <= curve < self.ncg_hesstol:
                    cg_break = 2
                    break
                # Standard CG iterations
                alpha = err / curve
                dx += alpha * p             # update solution
                res -= alpha * Hk_p      # update residual
                cg_iter +=1
                
                z = res.primal()            # update solver iterates
                
                
                err, old_err  =  res.apply(z),  err
                # check for CG convergence
                if err < cg_tol:
                    break
                
                beta  =  err / old_err
                p = z + beta * p
            
            xk, alpha = self.do_linesearch(objective, xk, dx)
            opt_iter +=1
            if cg_break != 0:
                break
        self.display(opt_iter, None, None, res)
        sol = {'Optimizer': xk,
               'Number of iterations': opt_iter,
               'Functional value at optimizer': None }
        return sol
                
 
