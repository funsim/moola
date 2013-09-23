from moola.linesearch import FixedLineSearch, ArmijoLineSearch, StrongWolfeLineSearch 
from moola.problem import Solution

class SteepestDescent(object):
    """
        Implements the steepest descent method. 
     """
    def __init__(self, tol=1e-4, options={}, **args):
        '''
        Initialises the steepest descent algorithm. The valid options are:
         * tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: armijo
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
          '''

        # Set the default options values
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", True)
        self.line_search = options.get("line_search", "armijo")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)

    def __str__(self):
        s = "Steepest descent method."
        s += "Line search:\t %s" % self.line_search 
        s += "Maximum iterations:\t %i" % self.maxiter 

    def solve(self, problem):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''

        j = None 
        j_prev = None
        dj = None 
        s = None

        m_prev = m.deep_copy()

        # Start the optimisation loop
        it = 0
        while True:

            # Evaluate the functional at the current iterate
            if j == None:
                j = J(m)
            dj = dJ()
            # TODO: Instead of reevaluating the gradient, we should just project dj 
            s = dJ(project=True) # The search direction is the Riesz representation of the gradient
            s.scale(-1)

            if disp:
                if MPI.process_number()==0: 
                    print "Iteration %i\tJ = %s\t|dJ| = %s" % (it, j, s.normL2())

            # Check for convergence                                                              # Reason:
            if not ((gtol    == None or s == None or s.normL2() > gtol) and                      # ||\nabla j|| < gtol
                    (tol     == None or j == None or j_prev == None or abs(j-j_prev)) > tol and  # \Delta j < tol
                    (maxiter == None or it < maxiter)):                                          # maximum iteration reached
                break

            # Compute slope at current point
            djs = dj.inner(s) 

            if djs >= 0:
                raise RuntimeError, "Negative gradient is not a descent direction. Is your gradient correct?" 

            # Define the real-valued reduced function in the s-direction 
            def phi(alpha):
                m.assign(m_prev)
                m.axpy(alpha, s) # m = m_prev + alpha*s

                return J(m)

            def phi_dphi(alpha):
                p = phi(alpha) 
                dj = dJ()
                djs = dj.inner(s) 
                return p, djs

            alpha = ls.search(phi, phi_dphi)

            # update m and j_new
            j_new = phi(alpha)

            # Update the current iterate
            m_prev.assign(m)
            j_prev = j
            j = j_new
            it += 1

            if "callback" in options:
                options["callback"](j, s, m)

        # Print the reason for convergence
        if disp:
            n = s.normL2()
            if MPI.process_number()==0:
                if maxiter != None and iter <= maxiter:
                    print "\nMaximum number of iterations reached.\n"
                elif gtol != None and n <= gtol: 
                    print "\nTolerance reached: |dJ| < gtol in %i iterations.\n" % it
                elif tol != None and j_prev != None and abs(j-j_prev) <= tol:
                    print "\nTolerance reached: |delta j| < tol in %i interations.\n" % it

        return m, {"Number of iterations": it}

def get_line_search_method(line_search, line_search_options):
    if line_search == "strong_wolfe":
        ls = StrongWolfeLineSearch(**line_search_options)
    elif line_search == "armijo":
        ls = ArmijoLineSearch(**line_search_options)
    elif line_search == "fixed":
        ls = FixedLineSearch(**line_search_options)
    else:
        raise ValueError, "Unknown line search specified. Valid values are 'armijo', 'strong_wolfe' and 'fixed'."

    return ls
