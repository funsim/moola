from .line_search import LineSearch
from .hz_linesearch import hz_linesearch

class HagerZhangLineSearch(LineSearch):
    def __init__(self, ftol=1e-4, gtol=0.9, xtol=1e-1, start_stp=1.0, stpmax=float('Inf'), display=False):
        '''
        This class implements a line search algorithm whose steps 
        satisfy the approximate Wolfe conditions
        '''

        self.ftol            = ftol 
        self.gtol            = gtol
        self.xtol            = xtol
        self.start_stp       = start_stp
        self.stpmax          = stpmax
        self.display         = display

    def search(self, phi, phi_dphi, phi_dphi0):
        ''' Performs the line search on the function phi. 

            phi must be a function [0, oo] -> R.
            phi_dphi must evaluate phi and its derivative, and 
            must be a function [0, oo] -> (R, R).

            The return value is a step that satisfies the strong Wolfe condition. 
        '''
            

        class DF(object):
            pass

        df = DF()
        df.phi = phi
        df.phi_dphi = phi_dphi

        mayterminate=True

        stp = hz_linesearch(df, c=self.start_stp, 
                mayterminate=mayterminate, 
                display=self.display,
                alphamax=self.stpmax)
        return stp
