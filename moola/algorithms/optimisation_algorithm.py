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
    
    def do_linesearch(self, obj, m, s):
        ''' Performs a linesearch on obj starting from m in direction s. '''

        m_new = m.copy()

        def update_m_new(alpha):
            if update_m_new.alpha_new != alpha:
                m_new.assign(m)
                m_new.axpy(alpha, s) 
                update_m_new.alpha_new = alpha
        update_m_new.alpha_new = 0

        # Define the real-valued functions in the s-direction 
        def phi(alpha):
            update_m_new(alpha)

            return obj(m_new)

        def phi_dphi(alpha):
            update_m_new(alpha)

            p = obj(m_new)
            djs = obj.derivative(m_new).apply(s)
            return p, djs

        # Perform the line search
        alpha = self.ls.search(phi, phi_dphi)

        update_m_new(alpha)
        return m_new, float(alpha)

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
            print(msg.format(it, J, grad.primal_norm(), deltaJ) + reasons[self.status] )
            
    
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


