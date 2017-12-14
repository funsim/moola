from .line_search import LineSearch
from .dcsrch import dcsrch
from numpy import zeros
from logging import error

class StrongWolfeLineSearch(LineSearch):
    def __init__(self, ftol=1e-4, gtol=0.9, xtol=1e-1, start_stp=1.0, stpmin = None, stpmax="automatic", verify=False, ignore_warnings=False):
        '''
        This class implements a line search algorithm whose steps
        satisfy the strong Wolfe conditions (i.e. they satisfies a
        sufficient decrease condition and a curvature condition).

        The algorithm is designed to find a step 'stp' that satisfies
        the sufficient decrease condition

               f(stp) <= f(0) + ftol*stp*f'(0),

        and the curvature condition

               abs(f'(stp)) <= gtol*abs(f'(0)).

        If ftol is less than gtol and if, for example, the function
        is bounded below, then there is always a step which satisfies
        both conditions.

        If no step can be found that satisfies both conditions, then
        the algorithm stops with a warning. In this case stp only
        satisfies the sufficient decrease condition.

        The function arguments are:

           ftol           | a nonnegative tolerance for the sufficient decrease condition.
           gtol           | a nonnegative tolerance for the curvature condition.
           xtol           | a nonnegative relative tolerance for an acceptable step.
           start_stp      | a guess for an initial step size.
           stpmin         | a nonnegative lower bound for the step.
           stpmax         | a nonnegative upper bound for the step.
           verify         | if True, the step is compared to the Fortran implementation
                          | (note: this assumes that the Python module dcsrch_fortran is compiled
                          |  and in the PYTHONPATH)
          ignore_warnings | Continue if the line search ends with a warnings (e.g. stp = stpmax).
                          | Default: True

        References:
         Mor'e, J.J., and Thuente, D.J., 1992, Line search algorithms with guaranteed
              sufficient decrease: Preprint MCS-P330-1092, Argonne National Laboratory.
         Averick, B.M., and Mor'e, J.J., 1993, FORTRAN subroutines dcstep and dcsrch
              from MINPACK-2, 1993, Argonne National Laboratory and University of Minnesota.
        '''

        self.ftol            = ftol
        self.gtol            = gtol
        self.xtol            = xtol
        self.start_stp       = start_stp
        self.stpmin          = stpmin
        self.stpmax          = stpmax
        self.verify          = verify
        self.ignore_warnings = ignore_warnings

    def search(self, phi, phi_dphi, phi_dphi0):
        ''' Performs the line search on the function phi.

            phi must be a function [0, oo] -> R.
            phi_dphi must evaluate phi and its derivative, and
            must be a function [0, oo] -> (R, R).

            The return value is a step that satisfies the strong Wolfe condition.
        '''

        # Set up the variables for dscrch
        isave = zeros(3)
        dsave = zeros(14)
        task = "START"

        f, g = phi_dphi0

        # Compute an estimate for the maximum step size
        if not self.stpmin:
            self.stpmin = 0.
        if self.stpmax == "automatic":
            stpmax = max(4*min(self.start_stp, 1.0), 0.1*f/(-g*self.ftol))
            print("-g", -g)
        else:
            stpmax = self.stpmax

        stp = min(self.start_stp, stpmax)

        while True:
            stp, task, isave, dsave = self.__csrch__(f, g, stp, task, isave, dsave, stpmax)

            if task in ("START", "FG"):
                f, g = phi_dphi(stp)
            else:
                break

        if "Error" in task:
            raise RuntimeError(task)
        elif "Warning" in task:
            if not self.ignore_warnings:
                raise Warning(task)
            else:
                print("Warning in line search: %s." % task.replace("Warning: ", ""))
                return stp
        else:
            assert task=="Convergence" or ("Warning" in task and self.ignore_warnings)

            if self.verify:
                # Recompute the step with the Fortran implementation and compare
                stp_fort = self.search_fortran(phi, phi_dphi, stpmax)
                if stp_fort is not None and stp_fort != stp:
                    raise RuntimeError("The line search verification failed!")

            return stp

    def search_fortran(self, phi, phi_dphi, stpmax):
        ''' Performs the line search on the function phi using the Fortran implementation
            of the line search algorithm.

            phi must be a function [0, oo] -> R.
            phi_dphi must evaluate phi and its derivative, and
            must be a function [0, oo] -> (R, R).

            The return value is a step that satisfies the strong Wolfe condition.
        '''

        try:
            import pyswolfe
            dphi = lambda x: phi_dphi(x)[1]
            ls_fort = pyswolfe.StrongWolfeLineSearch(phi(0), dphi(0), 1.0, phi, dphi, gtol=self.gtol,
                                                     xtol=self.xtol, ftol=self.ftol, stp=self.start_stp,
                                                     stpmin=self.stpmin, stpmax=stpmax)
            ls_fort.search()
            return ls_fort.stp

        except ImportError:
            error("The line search could not be verified. Did you compile the pyswolfe Fortran module?")

    def __csrch__(self, f, g, stp, task, isave, dsave, stpmax):
        stp, task, isave, dsave = dcsrch(stp, f, g, self.ftol, self.gtol, self.xtol, task, self.stpmin, stpmax, isave, dsave)
        return stp, task, isave, dsave

