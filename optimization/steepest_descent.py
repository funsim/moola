from dolfin import MPI, inner, assemble, dx, Function
from data_structures import CoefficientList, OptFunctional
from line_search import FixedLineSearch, ArmijoLineSearch, StrongWolfeLineSearch 
import numpy

def minimize_steepest_descent(rf, tol=1e-16, options={}, **args):
    """
        Implements the steepest descent method to minimise a functional.

        Arguments:
         * rf: the reduced functional to be minimised
         * tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: armijo
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.

        Return value:
          * m: The optimised control values. 
          * d: A dictionary with additional information about the optimisation run. It contains:
             "Number of iterations": The number of iterations performed. 

     """


    # Set the default options values
    gtol = options.get("gtol", 1e-4)
    maxiter = options.get("maxiter", 200)
    disp = options.get("disp", True)
    line_search = options.get("line_search", "armijo")
    line_search_options = options.get("line_search_options", {})

    if disp and MPI.process_number()==0:
        print "Optimising using steepest descent with a " + line_search + " line search." 
        print "Maximum optimisation iterations: %i" % maxiter 

    ls = get_line_search(line_search, line_search_options)

    of = OptFunctional(rf)
    m = of.m()
    m_prev = m.deep_copy()
    
    J =  of.j
    dJ = of.dj

    j = None 
    j_prev = None
    dj = None 
    s = None

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

def get_line_search(line_search, line_search_options):
    if line_search == "strong_wolfe":
        ls = StrongWolfeLineSearch(**line_search_options)
    elif line_search == "armijo":
        ls = ArmijoLineSearch(**line_search_options)
    elif line_search == "fixed":
        ls = FixedLineSearch(**line_search_options)
    else:
        raise ValueError, "Unknown line search specified. Valid values are 'armijo', 'strong_wolfe' and 'fixed'."

    return ls
