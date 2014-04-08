from optimisation_algorithm import *
from IPython import embed
class NewtonCG(OptimisationAlgorithm):
    ''' 
    An implementation of the Fletcher-Reeves method 
    described in Wright 2006, section 5.2. 
    '''
    def __init__(self, tol=None, options={}, **args):
        '''
        Initialises the Newton CG method. The valid options are:
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
	    print 'tol argument not supported. Will be ignored.'
	self.tol = tol
        self.gtol = options.get("gtol", 1e-4)
        self.maxiter = options.get("maxiter", 200)
        self.disp = options.get("disp", 2)
        self.line_search = options.get("line_search", "strong_wolfe")
        self.line_search_options = options.get("line_search_options", {})
        self.ls = get_line_search_method(self.line_search, self.line_search_options)
        self.callback = options.get("callback", None)

    def __str__(self):
        s = "Newton CG method.\n"
        s += "-"*30 + "\n"
        s += "Line search:\t\t %s\n" % self.line_search 
        s += "Maximum iterations:\t %i\n" % self.maxiter 
        return s

    def solve(self, problem, xk):
        ''' Solves the optimisation problem with the Fletcher-Reeves method. 
            Arguments:
             * problem: The optimisation problem.

            Return value:
              * solution: The solution to the optimisation problem 
         '''
        print self
        obj = problem.obj
        it = 0
        while True:
            if it>= self.maxiter:
                break
            dJ = obj.derivative(xk)

            rj = dJ.copy()
            rr = rj.primal_norm()
            
            ej = min( 0.5, rr**.5) * rr
            
            zj = dJ.primal()
            zj.scale(0.) # z0 = 0
            dj = rj.primal()
            dj.scale(-1.)
            iit = 0
            if self.check_convergence(it,None,None, rj):
                break
            self.display(it, None, None, rj)
            while True:
                if iit >= 100:
                    print 'maximum of {} inner iterations reached'.format(iit)
                    break
                Bj = obj.hessian(xk)
                Bjdj = Bj(dj)
                t = Bjdj.apply(dj)
                if t<= 0:
                    if iit == 0:
                        pk = -dJ.primal()
                        break
                    else:
                        pk = zj
                        break
                alphj = rr**2/ t #?
                zj = zj +alphj * dj    #zj.axpy(alphj, dj)
                rj = rj +alphj * Bjdj #rj.axpy(alphj, bj)
                rr, rr_old = rj.primal_norm(), rr
                print 'iit = {}\trr = {}\tej = {} '.format(iit, rr, ej)
                if rr < ej:
                    pk = zj
                    break
                betaj = rr / rr_old
                #dj.axpy(betaj, dj) # doesn't give expected result?
                #dj.axpy(-1., rj)
                dj = betaj * dj -rj.primal()
                iit +=1
            #
            xk, alpha = self.do_linesearch(obj, xk, pk)
            it +=1

        self.display(it, None, None, rj)
        sol = {'Optimizer': xk,
               'Number of iterations': it,
               'Functional value at optimizer': None }
        return sol
                
 
