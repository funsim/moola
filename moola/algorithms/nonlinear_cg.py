from optimisation_algorithm import *

class NonLinearCG(OptimisationAlgorithm):
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
        self.disp = options.get("disp", 2)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {'gtol':1e-1, 'xtol':1e-16})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)

        # method-specific settings
        self.cg_scheme = options.get("cg_scheme", 'HZ')

    def __str__(self):
        s = "Nonlinear CG ({}).\n".format(self.cg_scheme)
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def do_linesearch(self, obj, m, s):
        ''' Performs a linesearch on obj starting from m in direction s. '''

        # Define the real-valued reduced function in the s-direction 
        def phi(alpha):
            tmpm = m.copy()
            tmpm.axpy(alpha, s) 

            return obj(tmpm)

        def phi_dphi(alpha):
            tmpm = m.copy()
            tmpm.axpy(alpha, s) 

            p = phi(alpha) 
            djs = obj.derivative(tmpm).apply(s)
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
        dj_grad = dj.primal()
        

        s = dj_grad.copy() # search direction
        s.scale(-1)

        # Start the optimisation loop
        it = 0

        while self.check_convergence(it, None, None, dj_grad) == 0:
            self.display(it, None, None, dj_grad)
            # Perform the line search

            try:
                alpha = self.do_linesearch(obj, m, s)
            except (RuntimeError, Warning), e:
                self.status = 4
                self.reason = e.message
                break
            
            # Update m
            m_prev = m.copy()
            m.axpy(alpha, s)

            # Compute the relaxation value
            if self.cg_scheme == 'FR':
                b_old = dj.apply(dj_grad)
                dj = obj.derivative(m)
                dj_grad = dj.primal()
                b = dj.apply(dj_grad)
                beta = b/b_old
            if self.cg_scheme == 'HS':
                y = dj
                dj = obj.derivative(m)
                dj_grad = dj.primal()
                y.axpy(-1.,dj)
                beta = y.apply(dj_grad) /y(s)
            if self.cg_scheme in ('PR', 'PR+'):
                b = dj.apply(dj_grad)
                y = dj
                dj = obj.derivative(m)
                dj_grad = dj.primal()
                y.axpy(-1.,dj)
                a = y.apply(dj_grad)
                beta = -a /b
                if self.cg_scheme == 'PR+':
                    beta = max(beta,0)
            if self.cg_scheme == 'HZ':
                y = dj
                z = dj_grad
                dj = obj.derivative(m)
                dj_grad = dj.primal()
                y.axpy(-1.,dj)
                z.axpy(-1., dj_grad)
                a = y.apply(z)
                b = -y.apply(s)
                c = -y.apply(dj_grad)
                d = dj.apply(s)
                beta = (c - 2*d*a/b)/b
            if self.cg_scheme == 'DY':
                y = dj
                dj = obj.derivative(m)
                dj_grad = dj.primal()
                y.axpy(-1.,dj)
                beta = - dj.apply(dj_grad) / y(s)
            if self.cg_scheme == 'D':
                h = obj.hessian(m_prev, s)
                dj_grad = obj.derivative().primal()
                beta =  h.apply(dj_grad) / h(s)
                
            # Update the search direction
            s.scale(beta)
            s.axpy(-1, dj_grad)

            it += 1

            if self.callback is not None:
                self.callback(None, s, m)

        # Print the reason for convergence
        self.display(it, None, None, dj_grad)
        sol = Solution({"Optimizer": m,
                            "Number of iterations": it})
        return sol
