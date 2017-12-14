from .optimisation_algorithm import *
from .bfgs import LimitedMemoryInverseHessian, LinearOperator
from numpy import sqrt
from IPython import embed

def dual_to_primal(x):
    return x.primal()

class HybridCG(OptimisationAlgorithm):
    ''' 
    A Newton-CG method that falls back to L-BFGS when the curvature is non-positive.
    '''
    __name__ = 'NewtonCG'
    def __init__(self, problem, initial_point = None, Hinit=LinearOperator(dual_to_primal), options = {}):
        '''
        Initialises the Hybrid CG method. The valid options are:
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
        self.problem = problem
        self.set_options(options)
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'   : initial_point,
                     'iteration' : 0,
                     'lbfgs'     : LimitedMemoryInverseHessian(Hinit, self.options['mem_lim']) }
    
    def __str__(self):
        s = "Hybrid L-FBGS / CG method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search']
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s
    
    # set default parameters
    
    @classmethod
    def default_options(cls):
        # this is defined as a function to prevent defaults from being changed at runtime.
        default = OptimisationAlgorithm.default_options()
        default.update(
            # generic parameters:
            {"jtol"                   : None,
             "gtol"                   : 1e-4,
             "maxiter"                :  200,
             "display"                :    2,
             "line_search"            : "strong_wolfe",
             "line_search_options"    : {},
             "callback"               : None,
             "record"                 : ("grad_norm", "objective"),

             # method specific parameters:
             "initial_bfgs_iterations": 0,
             "ncg_reltol"             :  .5,
             "ncg_maxiter"            : 200,
             "ncg_hesstol"            : "default",
             "mem_lim"                : 5,
             "Hinit"                  : "default",
             })
        return default

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
        
        
        B = self.data['lbfgs']
        x = self.data['control']
        i = self.data['iteration']

        # compute initial objective and gradient
        J = objective(x)
        r = objective.derivative(x)  # initial residual ( with dk = 0)
        r.scale(-1.)
        self.update({'objective' : J,
                     'grad_norm' : r.primal_norm()})
        self.record_progress()
        
        if options['ncg_hesstol'] == "default":
            import numpy
            eps = numpy.finfo(numpy.float64).eps
            ncg_hesstol = eps*numpy.sqrt(len(x))
        else:
            ncg_hesstol = options['ncg_hesstol']
        
        # Start the optimisation loop
        while self.check_convergence() == 0:
            self.display(self.iter_status, 2)
            r0 = r.copy()
            p = Br = (B * r) # mapping residual to primal space
            d = p.copy().zero()
            rBr = r.apply(Br)
            H = objective.hessian(x)
            
            
            # CG iterations
            cg_tol =  min(options['ncg_reltol']**2, sqrt(rBr))*rBr 
            cg_iter  = 0
            cg_break = 0
            
            while cg_iter < options['ncg_maxiter'] and rBr >= cg_tol:
                if i < options['initial_bfgs_iterations']:
                    d = Br
                    break
                Hp  = H(p)
                pHp = Hp.apply(p)
                
                self.display('cg_iter = {}\tcurve = {}\thesstol = {}'.format(cg_iter, pHp, ncg_hesstol), 3)
                if pHp < 0:
                    #print 'TEST: not descent direction'
                    if cg_iter == 0:
                        # Fall back to L-BFGS update
                        d = Br
                        pass
                    # otherwise use the last computed pk
                    break
                        
                if 0 <= pHp < ncg_hesstol:
                    #cg_break = 2
                    if cg_iter == 0:
                        d = Br
                    # try to use what we have
                    try:
                        self.do_linesearch(objective, x, d) #TODO: fix this hack
                        #print 'TEST: below curvature treshold'
                        break
                    except:
                        pass
                # Standard CG iterations
                alpha = rBr / pHp
                d.axpy(alpha, p)            # update cg iterate
                r.axpy(-alpha, Hp)          # update residual

                Br = B*r
                t = r.apply(Br)
                rBr, beta = t, t / rBr, 

                p.scale(beta)
                p.axpy(1., Br)
                
                cg_iter +=1

            
            # do a line search and update
            x, a = self.do_linesearch(objective, x, d)
            d.scale(a)
            
            
            J, oldJ = objective(x), J

            # evaluate gradient at the new point
            r, r0 = objective.derivative(x), r0
            r.scale(-1)
            y = r0 - r
            
            # update the approximate Hessian
            
            B.update(y, d)
            i += 1

            # store current iteration variables
            self.update({'iteration' : i,
                         'control'   : x,
                         'grad_norm' : r.primal_norm(),
                         'delta_J'   : oldJ-J,
                         'objective' : J,
                         'lbfgs'     : B })
            self.record_progress()
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        return self.data
    
    
    
                
 
