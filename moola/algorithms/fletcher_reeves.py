from optimisation_algorithm import *
from numpy import sqrt
class FletcherReeves(OptimisationAlgorithm):
    ''' 
    An implementation of the Fletcher-Reeves method 
    described in Wright 2006, section 5.2. 
    '''
    def __init__(self, options={}, **args):
        '''
        Initialises the Fletcher-Reeves mehtod. The valid options are:
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Not supported yet - must be None. 
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
          '''

        # Set the default options values

        self.tol = options.get("tol", None)
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", 2)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.linesearch = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)

        if self.tol is not None:
            print 'Warning: tol argument not supported. Will be ignored.'

    def __str__(self):
        s = "Fletcher Reeves method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def solve(self, problem, m):
        ''' Solves the optimisation problem with the Fletcher-Reeves method. 
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        print self
        obj = problem.obj

        dj = obj.derivative(m)
        dj_grad = dj.primal()
        b = dj.apply(dj_grad)

        s = dj_grad.copy() # search direction
        s.scale(-1)

        # Start the optimisation loop
        it = 0

        while True:
            if it >= self.maxiter or sqrt(b) < self.gtol:
                break
            print "it = {}\t|grad J| = {}".format(it, sqrt(b)) 
            #self.display(it, None, None, dj_grad)
            # Perform the line search
            m, alpha = self.do_linesearch(obj, m, s)

            # Reevaluate the gradient
            dj = obj.derivative(m)
            dj_grad = dj.primal()

            # Compute the relaxation value
            b_old = b 
            b = dj.apply(dj_grad)
            beta = b/b_old

            # Update the search direction
            s.scale(beta)
            s.axpy(-1, dj_grad)

            it += 1

            if self.callback is not None:
                self.callback(None, s, m)

        # Print the reason for convergence
        print 'converged.'
        print "it = {}\t|grad J| = {}".format(it, sqrt(b)) 
        sol = Solution({"Optimizer": m,
                            "Number of iterations": it})
        return sol
