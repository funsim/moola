from optimisation_algorithm import *

class BFGSOperator(object):

    def __init__(self, sk, yk, Hk):
        self.sk = sk
        self.yk = yk
        self.Hk = Hk

    def __call__(self, d):
        sk = self.sk 
        yk = self.yk 
        Hk = self.Hk

        rhok = 1. / yk(sk)

        a1 = Hk( d - rhok * d(sk) * yk )
        a2 = a1 - rhok * ( yk(a1) ) * sk 
        a3 = rhok * d(sk) * sk

        return a2 + a3


class RieszMap(object):

    def __call__(self, d): 
        return d.riesz_representation()
        

class BFGS(OptimisationAlgorithm):
    """
        Implements the BFGS method. 
     """
    def __init__(self, tol=1e-4, H_init=RieszMap(), options={}, hooks={}, **args):
        '''
        Initialises the steepest descent algorithm. 
        
        Valid options are:
        
         * tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
         * H_init: Initial approximation of the inverse Hessian.
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
         * hooks: A dictionariy containing user-defined "hook" functions that are called at certain events during the optimisation.
            - before_iteration: Is called after before each iteration.
            - after_iteration: Is called after each each iteration.
          '''

        # Set the default options values
        self.tol = tol
        self.H_init = H_init
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", True)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)
        self.hooks = hooks

    def __str__(self):
        s = "BFGS method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def display(self):
        print "disp be written"


    def check_convergence(self):
        print "check_convergergence to be written"
        return False, ""

    def perform_line_search(self, xk, pk):
        print "perform_line_search to be written"
        return 1.

    def solve(self, problem, x_init):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        obj = problem.obj

        Hk = self.H_init
        xk = x_init.copy()
        dJ_old = obj.derivative(xk)

        # Start the optimisation loop
        it = 0
        while True:
            #hook("before_iteration", j, grad)
            self.display()

            conv, reason = self.check_convergence()
            if conv is True:
                break
            
            # compute search direction
            pk = - Hk(dJ_old)
            
            # do a line search
            ak = self.perform_line_search(xk, pk)
            
            sk = ak * pk
            xk += sk

            # evaluate gradient at the new point
            dJ = obj.derivative(xk)
            yk = dJ - dJ_old
            
            
            # update the approximate Hessian
            Hk = BFGSOperator(sk, yk, Hk)
            
            dJ_old = dJ
            it += 1

            #hook("after_iteration", j, grad)

        self.display()

        return xk

