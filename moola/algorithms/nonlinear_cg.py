from .optimisation_algorithm import *
from numpy import sqrt

class NonLinearCG(OptimisationAlgorithm):
    ''' 
    An implementation of the Fletcher-Reeves method 
    described in Wright 2006, section 5.2. 
    '''
    def __init__(self, problem, initial_point = None, options={}, **args):
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
        self.problem = problem
        self.set_options(options)
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'      : initial_point,
                     'iteration'    : 0}
        self.compute_beta = _beta_rules[self.options['beta_rule']]
    
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
             "beta_rule"              : "polak-ribiere-polyak"
             })
        return default

    @classmethod
    def list_beta_rules(cls):
        return list(_beta_rules.keys())
    

    def __str__(self):
        s = "Nonlinear CG ({}).\n".format(self.options['beta_rule'])
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.options['line_search']
        s += "Maximum iterations:\t %i\n" % self.options['maxiter']
        return s
    
    def solve(self):
        ''' Solves the optimisation problem with a (conjugate) gradient method. 
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        self.display(self.__str__(), 1)

        options = self.options
        obj = self.problem.obj

        x = self.data['control']

        J = obj(x)
        derivative = obj.derivative(x)
        gradient = derivative.primal()
        grad_norm2 = derivative.apply(gradient)

        self.update({'grad_norm' : sqrt(grad_norm2),
                     'objective' : obj(x)})
        self.record_progress()
        
        p = gradient.copy() # initial search direction
        p.scale(-1)

        # Start the optimisation loop
        it = 0
        while self.check_convergence() == 0:
            self.display(self.iter_status, 2)

            # Perform the line search
            try:
                x, alpha = self.do_linesearch(obj, x, p)
            except RuntimeError:
                # TODO: handle errors better.
                # For assuming error is due to search diretion not being a descent direction.
                self.display('Search direction is not a descent direction; restarting.', 2)
                p = -1 * gradient
                x, alpha = self.do_linesearch(obj, x, p)
                

            # gradient evaluations at new point
            derivative, old_derivative = obj.derivative(x), derivative 
            gradient, old_gradient = derivative.primal(), gradient
            grad_norm2, old_grad_norm2 = derivative.apply(gradient), grad_norm2
            
            # Compute the relaxation value
            beta = self.compute_beta(gradient, derivative, grad_norm2, p, 
                                     old_gradient, old_derivative, old_grad_norm2)

            # get new search direction:  p_new = beta * p_old - gradient
            p.scale(beta)
            p.axpy(-1., gradient)
            it += 1
            J, old_J = obj(x), J
            #update
            self.update(iteration = it, control = x, grad_norm = sqrt(grad_norm2),
                        objective = J, delta_J = old_J - J)
            self.record_progress()

        # Print the reason for convergence
        self.display(self.convergence_status, 1)
        self.display(self.iter_status, 1)
        
        return self.data



    
def _steepest_descent(gradient, derivative, grad_norm2, p,
                      old_gradient, old_derivative, old_grad_norm2):
    return 0.

def _fletcher_reeves(gradient, derivative, grad_norm2, p,
                     old_gradient, old_derivative, old_grad_norm2):
    '''
    Compute the Flether-Reeves update parameter.
    '''
    beta = grad_norm2 / old_grad_norm2
    return beta

def _hestenes_stiefel(gradient, derivative, grad_norm2, p,
                      old_gradient, old_derivative, old_grad_norm2):
    '''
    Computes the 'standard cg' beta 
    '''
    y = derivative - old_derivative
    beta = y.apply(gradient) / y.apply(p)
    return beta

def  _conjugate_descent(gradient, derivative, grad_norm2, p,
                        old_gradient, old_derivative, old_grad_norm2):
    beta = grad_norm2 / - old_derivative.apply(p)
    return beta

def  _polak_ribiere_polyak(gradient, derivative, grad_norm2, p,
                    old_gradient, old_derivative, old_grad_norm2):
    y = derivative - old_derivative
    beta = y.apply(gradient) / old_grad_norm2
    return beta

def _hager_zhang(gradient, derivative, grad_norm2, p,
                 old_gradient, old_derivative, old_grad_norm2):
    y = derivative - old_derivative
    z = y.primal()
    t1 = y.apply(p)
    t2 = 2 * y.apply(z) / t1
    beta = derivative.apply(z - t2*p) / t1
    return beta

def _dai_yuan(gradient, derivative, grad_norm2, p,
              old_gradient, old_derivative, old_grad_norm2):
    y = derivative - old_derivative
    beta = grad_norm2 / y.apply(p)
    return beta

def _liu_storey(gradient, derivative, grad_norm2, p,
                      old_gradient, old_derivative, old_grad_norm2):
    y = derivative - old_derivative
    return - y.apply(gradient) /  old_derivative.apply(p)
    

_beta_rules = {'steepest_descent'    : _steepest_descent,
               'fletcher-reeves'     : _fletcher_reeves,
               'hestenes-stiefel'    : _hestenes_stiefel,
               'conjugate_descent'   : _conjugate_descent,
               'polak-ribiere-polyak': _polak_ribiere_polyak,
               'hager-zhang'         : _hager_zhang,
               'dai-yuan'            : _dai_yuan,
               'liu-storey'          : _liu_storey}

