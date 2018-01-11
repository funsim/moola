from .line_search import LineSearch

class ArmijoLineSearch(LineSearch):
    def __init__(self, ftol = 1e-4, start_stp = 1.0, stpmin = 1e-10, stpmax=10000,
                 adaptive_stp=True):
        '''
        This class implements a line search algorithm whose steps 
        satisfy the Armijo condition, i.e. they satisfies a 
        sufficient decrease condition.

        The algorithm is designed to find a step 'stp' that satisfies
        the sufficient decrease condition

               f(stp) <= f(0) + ftol*stp*f'(0).

        There is always a step which satisfies both conditions.

        If the step size drops below stpmin, the search algorithm raises Warning.

        The function arguments are:

           ftol         | a nonnegative tolerance for the sufficient decrease condition.
           start_stp    | a guess for an initial step size. 
           stpmin       | a nonnegative lower bound for the step.
           stpmax       | a nonnegative upper bound for the step. 
           adaptive_stp | dis/enables the adaptive step size algorithm.

        Returns:
          A step size that satisfies the Armijo condition

        '''

        if ftol <= 0:
            raise ValueError("ftol must be > 0")
        if start_stp <= 0:
            raise ValueError("start_stp must be > 0")
        if stpmin <= 0:
            raise ValueError("stpmin must be > 0")
        if stpmax <=0 or stpmax <= stpmin:
            raise ValueError("stpmax must be > 0 and > stpmin")
        
        self.ftol         = ftol 
        self.start_stp    = start_stp
        self.stpmin       = stpmin
        self.adaptive_stp = adaptive_stp
        self.stpmax       = stpmax

    def _test(self, f, g, f0, stp):
        ''' Tests if the Armijo condition is satisfied '''
        return f <= f0 + self.ftol*stp*g

    def _adapt(self, it):
        ''' Adapts the starting step according '''
        if it < 2:
            self.start_stp *= 2
        if it >=2:
            self.start_stp /= 2
        if self.start_stp >= self.stpmax:
            self.start_stp /=4

    def search(self, phi, phi_dphi, phi_dphi0):
        ''' Performs the line search on the function phi. 

            phi must be a function [0, oo] -> R.
            phi_dphi is ignored (you can set it to None).
            phi_dphi0 is a tuple containing the functional value and 
            directional derivative at step=0.

            The return value is a step that satisfies the Armijo condition. 
        '''
            
        stp = self.start_stp
        finit, ginit = phi_dphi0
        if ginit >= 0:
            raise Warning("The gradient is not a descent direction")
        f = finit 
        
        it = 0
        while True:
            if self._test(f, ginit, finit, stp):
                self._adapt(it)
                return stp
            elif stp < self.stpmin:
                raise Warning("The step size dropped below the minimum step size.")

            stp /= 2.0
            it += 1
            f = phi(stp)
