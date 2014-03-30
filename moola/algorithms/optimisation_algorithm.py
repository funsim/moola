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
