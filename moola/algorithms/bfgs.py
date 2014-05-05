from optimisation_algorithm import *


class LinearOperator(object):

    def __init__(self, matvec):
        self.matvec = matvec

    def __mul__(self,x):
        return self.matvec(x)

    def __rmul__(self,x):
        return NotImplemented

    def __call__(self,x):
        return self.matvec(x)


def dual_to_primal(x):
    return x.primal()
 

class LimitedMemoryInverseHessian(LinearOperator):
    '''
    This class implements the limit-memory BFGS approximation of the inverse Hessian.
    '''
    def __init__(self, Hinit, mem_lim = 10):
        self.Hinit = Hinit
        self.mem_lim = mem_lim
        self.y   = []
        self.s   = []
        self.rho = []
        
    def __len__(self):
        assert( len(self.y) == len(self.s) )
        return len(self.y)

    def __getitem__(self,k):
        if k==0:
            return self.Hinit
        return (self.rho[k-1], self.y[k-1], self.s[k-1])

    def update(self,yk, sk):
        if self.mem_lim == 0:
            return
        if len(self) == self.mem_lim:
            self.y   = self.y[1:]
            self.s   = self.s[1:]
            self.rho = self.rho[1:]
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append( 1./ yk.apply(sk) )

    def matvec(self,x,k = -1):
        if k == -1:
            k = len(self)
        if k == 0:
            return self.Hinit * x
        rhok, yk, sk = self[k]     
        t = x - rhok * x.apply(sk) * yk
        t = self.matvec(t, k-1)
        t = t - rhok * yk.apply(t) * sk
        t = t + rhok * x.apply(sk) * sk
        return t

class BFGS(OptimisationAlgorithm):
    """
        Implements the BFGS method. 
     """
    def __init__(self, problem, initial_point = None, Hinit=LinearOperator(dual_to_primal), options={}, hooks={}, **args):
        '''
        Initialises the L-BFGS algorithm. 
        
        Valid options are:
        
         * Hinit: Initial approximation of the inverse Hessian.
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
         * hooks: A dictionariy containing user-defined "hook" functions that are called at certain events during the optimisation.
            - before_iteration: Is called after before each iteration.
            - after_iteration: Is called after each each iteration.
          '''
        # Set the default options values
        self.problem = problem
        self.set_options(options)
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'   : initial_point,
                     'iteration' : 0,
                     'lbfgs'     : LimitedMemoryInverseHessian(Hinit, self.options['mem_lim']) }
        
        

    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"jtol"                   : 1e-4,
             "gtol"                   : 1e-4,
             "maxiter"                :  200,
             "display"                :    2,
             "line_search"            : "strong_wolfe",
             "line_search_options"    : {},
             "callback"               : None,
             "record"                 : ("grad_norm", "objective"),
             # method specific parameters:
             "mem_lim"                : 5,
             "Hinit"                  : "default",
             })
        return default
    
    def __str__(self):
        s = "L-BFGS method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search'] 
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s

    def solve(self):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        self.display( self.__str__(), 1)

        objective = self.problem.obj
        options = self.options
        
        Hk = self.data['lbfgs']
        xk = self.data['control']
        it = self.data['iteration']

        # compute initial objective and gradient
        J = objective(xk)
        dJ_xk = objective.derivative(xk)
        self.update({'objective' : J,
                     'grad_norm' : dJ_xk.primal_norm()})
        self.record_progress()
        
        # Start the optimisation loop
        while self.check_convergence() == 0:
            self.display(self.iter_status, 2)
            # compute search direction
            pk = - (Hk * dJ_xk)
            
            # do a line search and update
            xk, ak = self.do_linesearch(objective, xk, pk)
            pk.scale(ak)
            
            J, oldJ = objective(xk), J

            # evaluate gradient at the new point
            dJ_xk, dJ_old = objective.derivative(xk), dJ_xk
            yk = dJ_xk - dJ_old
            
            # update the approximate Hessian
            Hk.update(yk, pk)

            it += 1

            # store current iteration variables
            self.update({'iteration' : it,
                         'control'   : xk,
                         'grad_norm' : dJ_xk.primal_norm(),
                         'delta_J'   : oldJ-J,
                         'objective' : J,
                         'lbfgs'     : Hk })
            self.record_progress()
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data

