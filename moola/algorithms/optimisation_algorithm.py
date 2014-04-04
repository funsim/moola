from moola.linesearch import FixedLineSearch, ArmijoLineSearch, StrongWolfeLineSearch 
from moola.problem import Solution

class OptimisationAlgorithm(object):
    '''
    An abstract implementation of an optimisation algorithm.
    All implementations of optimisation algorithms should use this is a base class.
    '''
    def __init__(self, tol, options={}, **args):
        ''' Initialises the steepest descent algorithm. '''
        raise NotImplementedError, 'OptimisationAlgorithm.__init__ is not implemented'

    def __str__(self):
        ''' Prints out a description of the algorithm settings. '''
        raise NotImplementedError, 'OptimisationAlgorithm.__str__ is not implemented'
    
    def perform_line_search(self, xk, dk):
        '''
        Performs a line search for a new minimum.
        xk is the current control point
        pk is the direction of the search

        This function should return a triple, consisting of:
        The computed stepsize ak
        The objective evaluated at xk + ak * dk
        The derivative at xk + ak * dk, if computed
        '''
        p_last = None
        djs = None
        
        def phi(alpha):
            tmpx = xk.copy()
            tmpx.axpy(alpha, pk)
            return self.problem.obj(tmpx)

        def phi_dphi(alpha):
            tmpx = xk.copy()
            tmpx.axpy(alpha, pk)
            p = p_last = phi(alpha) 
            djs = self.problem.obj.derivative(tmpx)(pk)
            return p, djs

        ak, Jnew, DJnew = self.ls.search(phi, phi_dphi)
        return float(ak), Jk, DJk # TODO: numpy.float64 does not play nice with moola types

    def check_convergence(self, it, J, oldJ, g):
        s = 0

        if it >= self.maxiter:
            s = 1

        elif g != None and g.primal_norm() < self.gtol != None:
            s = 3

        elif J != None and oldJ != None and self.tol != None:
            if J == oldJ == 0:
                s = 2
            elif abs(1 - min(J/oldJ, oldJ/J)) <= self.tol:
                s = 2


        self.status = s
        return self.status

    def display(self,it, J, oldJ, grad):
        deltaJ = oldJ -J if (oldJ is not None and J is not None) else None
        msg = 'Iteration {0:d}' \
            + ('\tJ = {1:e}' if J is not None else '')\
            + ('\t|dJ| = {2:e}' if grad is not None else '')\
            + ('\tdeltaJ = {3:e}' if deltaJ is not None else '') 
        if self.status == 0 and self.disp > 1:
            print(msg.format(it, J, grad.primal_norm(), deltaJ))
        if self.status != 0 and self.disp > 0:
            reasons = {1: '\nMaximum number of iterations reached.\n',
                       2: '\nTolerance reached: |delta j| < tol.\n',
                       3: '\nTolerance reached: |dJ| < gtol.\n',
                       4: 'Linesearch failed.',
                       5: 'Algorithm breakdown.'}
            print(reasons[self.status] +  msg.format(it, J, grad.primal_norm(), deltaJ))
            
    
    def solve(self, problem, m):
        ''' Solves the optimisation problem. 
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
        '''
        raise NotImplementedError, 'OptimisationAlgorithm.solve is not implemented'

def get_line_search_method(line_search, line_search_options):
    ''' Takes a name of a line search method and returns its Python implementation as a function. '''
    if line_search == "strong_wolfe":
        ls = StrongWolfeLineSearch(**line_search_options)
    elif line_search == "armijo":
        ls = ArmijoLineSearch(**line_search_options)
    elif line_search == "fixed":
        ls = FixedLineSearch(**line_search_options)
    else:
        raise ValueError, "Unknown line search specified. Valid values are 'armijo', 'strong_wolfe' and 'fixed'."

    return ls


