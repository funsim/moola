from optimisation_algorithm import *


class  LinearOperator(object):
    def __init__(self, matvec):
        self.matvec = matvec
    def __mul__(self,x):
        return self.matvec(x)
    def __rmul__(self,x):
        return NotImplemented
    def __call__(self,x):
        return self.matvec(x)

        

class LHess(LinearOperator):
    '''
    This class implements the limit-memory BFGS approximation of the inverse Hessian.
    '''
    def __init__(self, Hinit=1, mem_lim = 10):
        self.Hinit = Hinit
        self.mem_lim = mem_lim
        self.y   = []
        self.s   = []
        self.rho = []
    def __len__(self):
        assert( len(self.y) == len(self.s) )
        return len(self.y)+1
    def __getitem__(self,k):
        if k==0:
            return self.Hinit
        return (self.rho[k-1], self.y[k-1], self.s[k-1])
    def update(self,yk, sk):
        if self.mem_lim == 0:
            return
        if len(self) == self.mem_lim+1:
            self.y   = self.y[1:]
            self.s   = self.s[1:]
            self.rho = self.rho[1:]
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append( 1./ yk.dot(sk) )
    def matvec(self,x,k = -1):
        if k == -1:
            k = len(self)-1
        if k == 0:
            return self.Hinit * x
        rhok, yk, sk = self[k]     
        t = x - rhok * x.dot(sk) * yk
        t = self.matvec(t, k-1)
        t = t - rhok * yk.dot(t) * sk
        t = t + rhok * x.dot(sk) * sk
        return t

class BFGS(OptimisationAlgorithm):
    """
        Implements the BFGS method. 
     """
    def __init__(self, Hinit=1, options={}, hooks={}, **args):
        '''
        Initialises the steepest descent algorithm. 
        
        Valid options are:
        
         * tol: Functional reduction stopping tolerance: |j - j_prev| < tol. Default: 1e-4.
         * H_init: Initial approximation of the inverse Hessian.
         * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
            - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200. 
            - disp: dis/enable outputs to screen during the optimisation. Default: True
            - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
            - line_search: defines the line search algorithm to use. Default: strong_wolfe
            - line_search_options: additional options for the line search algorithm. The specific options read the help 
              for the line search algorithm.
            - an optional callback method which is called after every optimisation iteration.
         * hooks: A dictionariy containing user-defined "hook" functions that are called at certain events during the optimisation.
            - before_iteration: Is called after before each iteration.
            - after_iteration: Is called after each each iteration.
          '''

        # Set the default options values
        self.tol = options.get("tol", 1e-4)
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", 2)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)
        self.hooks = hooks

        # method-specific settings:
        self.Hinit = Hinit
        self.mem_lim  = options.get("mem_lim", 20)

    def __str__(self):
        s = "BFGS method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def perform_line_search(self, xk, pk):
        def phi(alpha):
            tmpx = xk.copy()
            tmpx.axpy(alpha, pk)
            return self.problem.obj(tmpx)

        def phi_dphi(alpha):
            tmpx = xk.copy()
            tmpx.axpy(alpha, pk)
            j = self.problem.obj(tmpx)
            djs = self.problem.obj.derivative(tmpx)(pk)
            return j, djs

        ak = self.ls.search(phi, phi_dphi)
        return float(ak) # numpy.float64 does not play nice with moola types
        

    def solve(self, problem, xinit):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        if self.disp>0 : print self
        self.problem = problem
        obj = problem.obj

        Hk = LHess(self.Hinit, mem_lim = self.mem_lim)
        xk = xinit.copy()
        dJ_old = obj.derivative(xk)
        J  = obj(xk)                 # TODO: combine in one call
        oldJ = None
        # Start the optimisation loop
        it = 0
        while True:
            #hook("before_iteration", j, grad)
            

            if self.check_convergence(it, J, oldJ, dJ_old) != 0:
                break
            self.display(it, J, oldJ, dJ_old)
            # compute search direction
            pk = - (Hk * dJ_old)

            # do a line search and update
            ak = self.perform_line_search(xk, pk)            
            sk = ak * pk
            xk += sk
            
            J, oldJ = obj(xk), J # FIXME: too many calls

            # evaluate gradient at the new point
            dJ = obj.derivative(xk)
            yk = dJ - dJ_old
            
            # update the approximate Hessian
            Hk.update(yk, sk)

            
            import numpy
            H = numpy.array([ Hk*numpy.array([1,0]), Hk*numpy.array([0,1])]).T
            l = numpy.linalg.eigvals(H)
            print l
            if l.min()  < 0:
                from IPython import embed; embed()
            dJ_old = dJ
            it += 1


        
        self.display(it, J, oldJ, dJ_old)
        sol =  {"Optimizer" :xk,
                "Number of iterations": it}
        #from IPython import embed; embed()
        return sol

