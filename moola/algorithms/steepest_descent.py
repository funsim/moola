from .optimisation_algorithm import *

class SteepestDescent(OptimisationAlgorithm):
    """
        Implements the steepest descent method.
     """
    def __init__(self, problem, initial_point=None, options={}, hooks={}, **args):
        '''
        Initialises the steepest descent algorithm.

        Valid options are:

         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
         * hooks: A dictionary containing user-defined "hook" functions that are called at certain events during the optimisation.
            - before_iteration: Is called after before each iteration.
            - after_iteration: Is called after each each iteration.
          '''

        # Set the default options values
        self.problem = problem
        self.set_options(options)
        self.hooks = hooks
        self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
        self.data = {'control'   : initial_point,
                     'iteration' : 0}

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
             "line_search_options"    : {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1},
             "record"                 : ("grad_norm", "objective"),
             })
        return default

    def __str__(self):
        s = "Steepest descent method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search
        s += "Maximum iterations:\t %i\n" % self.maxiter
        return s

    def solve(self):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem
         '''
        j = None
        j_prev = None

        m = self.data['control']
        obj = self.problem.obj

        # Start the optimisation loop
        it = 0
        while True:

            grad = obj.derivative(m).primal()
            self.update({'grad_norm' : grad.norm()})
            self.display(self.iter_status, 2)

            if "before_iteration" in self.hooks:
                self.hooks["before_iteration"](j, grad)

            # Check for convergence
            if self.check_convergence() != 0:
                break

            m, alpha = self.do_linesearch(obj, m, -grad)
            j_prev, j = j, obj(m)

            # Update the iterate counter
            it += 1

            if "after_iteration" in self.hooks:
                self.hooks["after_iteration"](j, grad)

            # Print the reason for convergence
            self.update({"control"   : m,
                         "iteration" : it,
                         "grad_norm" : grad.norm(),
                         "objective" : j})
            self.record_progress()
        return self.data
