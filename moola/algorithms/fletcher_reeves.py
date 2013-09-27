from optimisation_algorithm import *

class FletcherReeves(OptimisationAlgorithm):
    ''' 
    An implementation of the Fletcher-Reeves method 
    described in Wright 2006, section 5.2. 
    '''
    def __init__(self, tol=None, options={}, **args):
        '''
        Initialises the Fletcher-Reeves mehtod. The valid options are:
         * tol: Not supported - must be None. 
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
	    raise ValueError, 'tol argument not supported. Must be None'
	self.tol = tol
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", True)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)

    def __str__(self):
        s = "Fletcher Reeves method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def test_convergence(self, it, j, j_prev, dj, reason=False):
	''' Test if the optimisation iteration has converged. '''

	converged = True

	if self.gtol != None and dj != None and dj.norm("L2") < self.gtol:
	    reason = "Tolerance reached: |dJ| < gtol." 
	elif self.tol != None and j != None and j_prev != None and abs(j-j_prev) < self.tol:
            reason = "Tolerance reached: |delta j| < tol."
	elif self.maxiter != None and it > self.maxiter:
	    reason = "Maximum number of iterations reached."
	else:
	    reason = None
	    converged = False

	if reason:
	    return reason
	else:
	    return converged

    def do_linesearch(self, obj, m, s):
        ''' Performs a linesearch on obj starting from m in direction s. '''

	# Define the real-valued reduced function in the s-direction 
	def phi(alpha):
	    tmpm = m.__class__(m)
	    tmpm.axpy(alpha, s) 

	    return obj(tmpm)

	def phi_dphi(alpha):
	    tmpm = m.__class__(m)
	    tmpm.axpy(alpha, s) 

	    p = phi(alpha) 
	    djs = obj.derivative(tmpm)(s)
	    return p, djs

	# Perform the line search
	alpha = self.ls.search(phi, phi_dphi)
	return alpha

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
	dj_grad = obj.gradient(m)
	b = dj(dj_grad)

	s = dj_grad.__class__(dj_grad)
	s.scale(-1)

        # Start the optimisation loop
        it = 0

	while not self.test_convergence(it, None, None, dj_grad):
	    if self.disp:
		print "Iteration %i\t|dJ| = %s" % (it, dj_grad.norm("L2"))

	    # Perform the line search
	    alpha = self.do_linesearch(obj, m, s)

            # Update m
            m.axpy(alpha, s)

            # Reevaluate the gradient
	    dj = obj.derivative(m)
	    dj_grad = obj.gradient(m)
	    s = s.__class__(dj_grad)

            # Compute the relaxation value
            b_old = b 
	    b = dj(dj_grad)
	    beta = b/b_old

            # Update the search direction
	    s.scale(beta)
	    s.axpy(-1, dj_grad)

            it += 1

            if self.callback is not None:
                self.callback(None, s, m)

        # Print the reason for convergence
        print self.test_convergence(it, None, None, dj_grad, reason=True)

        sol = Solution({"Optimizer": m,
                        "Number of iterations": it})
        return sol
