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
    def __init__(self, Hinit=1, max_len = 10):
        self.Hinit = Hinit
        self.max_len = max_len
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
        if len(self) == self.max_len:
            self.y   = self.y[1:]
            self.s   = self.s[1:]
            self.rho = self.rho[1:]
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append( yk.dot(sk) )
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
        self.Hinit = Hinit
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", True)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)
        self.hooks = hooks

    def __str__(self):
        s = "BFGS method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def display(self):
        print "disp be written"


    def check_convergence(self, it, J, oldJ, g):
        reasons = {1: 'Maximum number of iterations reached.',
                   2: 'Relative gain below tol.',
                   3: 'Norm of the derivative below gtol.'}
        r = 0
        if it > self.maxiter:
            r = 1
        if J != None and oldJ != None and oldJ-J < self.tol != None:
            r = 2
        if g!= None and g.norm() < self.gtol != None:
            r = 3
        if r != 0:
            return True, reasons[r]
        return False, None

    def perform_line_search(self, xk, pk):
        def phi(alpha):
            tmpx = xk.__class__(xk)
            tmpx.axpy(alpha, pk)
            return self.problem.obj(tmpx)

        def phi_dphi(alpha):
            tmpx = xk.__class__(xk)
            tmpx.axpy(alpha, pk)
            p = phi(alpha) 
            djs = self.problem.obj.derivative(tmpx)(pk)
            return p, djs

        ak = self.ls.search(phi, phi_dphi)
        return float(ak) # numpy.float64 does not play nice with moola types
        

    def solve(self, problem, xinit):
        '''
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        self.problem = problem
        obj = problem.obj

        Hk = LHess(self.Hinit)
        xk = xinit.copy()
        dJ_old = obj.derivative(xk)
        J, oldJ= None, None
        # Start the optimisation loop
        it = 0
        while True:
            #hook("before_iteration", j, grad)
            self.display()

            conv, reason = self.check_convergence(it, J, oldJ, dJ_old)
            if conv is True:
                break
            
            # compute search direction
            pk = - (Hk * dJ_old)
            
            # do a line search and update
            ak = self.perform_line_search(xk, pk)            
            sk = ak * pk
            xk += sk
            J, oldJ = obj(xk), J # FIXME: this is horrible!

            # evaluate gradient at the new point
            dJ = obj.derivative(xk)
            yk = dJ - dJ_old
            
            # update the approximate Hessian
            Hk.update(yk, sk)

            dJ_old = dJ
            it += 1


        self.display()

        ret =  {"Optimizer" :xk,
                "Number of iterations": it}
        return ret

