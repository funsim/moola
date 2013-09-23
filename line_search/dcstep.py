def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """

     Subroutine dcstep

     This subroutine computes a safeguarded step for a search
     procedure and updates an interval that contains a step that
     satisfies a sufficient decrease and a curvature condition.

     The parameter stx contains the step with the least function
     value. If brackt is set to True: a minimizer has
     been bracketed in an interval with endpoints stx and sty.
     The parameter stp contains the current step.
     The subroutine assumes that if brackt is set to True:

           min(stx,sty) < stp < max(stx,sty),

     and that the derivative at stx is negative in the direction
     of the step.

     The subroutine statement is

       subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                         stpmin,stpmax)

     where

       stx is a double precision variable.
         On entry stx is the best step obtained so far and is an
            endpoint of the interval that contains the minimizer.
         On exit stx is the updated best step.

       fx is a double precision variable.
         On entry fx is the function at stx.
         On exit fx is the function at stx.

       dx is a double precision variable.
         On entry dx is the derivative of the function at
            stx. The derivative must be negative in the direction of
            the step, that is, dx and stp - stx must have opposite
            signs.
         On exit dx is the derivative of the function at stx.

       sty is a double precision variable.
         On entry sty is the second endpoint of the interval that
            contains the minimizer.
         On exit sty is the updated endpoint of the interval that
            contains the minimizer.

       fy is a double precision variable.
         On entry fy is the function at sty.
         On exit fy is the function at sty.

       dy is a double precision variable.
         On entry dy is the derivative of the function at sty.
         On exit dy is the derivative of the function at the exit sty.

       stp is a double precision variable.
         On entry stp is the current step. If brackt is set to True
           : on input stp must be between stx and sty.
         On exit stp is a new trial step.

       fp is a double precision variable.
         On entry fp is the function at stp
         On exit fp is unchanged.

       dp is a double precision variable.
         On entry dp is the the derivative of the function at stp.
         On exit dp is unchanged.

       brackt is an logical variable.
         On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to .false.
         On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to True

       stpmin is a double precision variable.
         On entry stpmin is a lower bound for the step.
         On exit stpmin is unchanged.

       stpmax is a double precision variable.
         On entry stpmax is an upper bound for the step.
         On exit stpmax is unchanged.

     MINPACK-1 Project. June 1983
     Argonne National Laboratory.
     Jorge J. More' and David J. Thuente.

     MINPACK-2 Project. November 1993.
     Argonne National Laboratory and University of Minnesota.
     Brett M. Averick and Jorge J. More'.

     Converted to Python and extended, July 2013
     Imperial College London
     Simon W. Funke
    """

    from math import sqrt

    sgnd = dp*(dx/abs(dx))

    #     First case: A higher function value. The minimum is bracketed.
    #     If the cubic step is closer to stx than the quadratic step, the
    #     cubic step is taken, otherwise the average of the cubic and
    #     quadratic steps is taken.
    if fp > fx:
        theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)**2-(dx/s)*(dp/s))
        if (stp < stx):
            gamma = -gamma
        p = (gamma-dx) + theta
        q = ((gamma-dx)+gamma) + dp
        r = p/q
        stpc = stx + r*(stp-stx)
        stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2.0)*(stp-stx)
        if abs(stpc-stx) < abs(stpq-stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq-stpc)/2.0
        brackt = True

    #     Second case: A lower function value and derivatives of opposite
    #     sign. The minimum is bracketed. If the cubic step is farther from
    #     stp than the secant step, the cubic step is taken, otherwise the
    #     secant step is taken.
    elif sgnd < 0:
        theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)**2-(dx/s)*(dp/s))

        if stp > stx: 
            gamma = -gamma
        p = (gamma-dp) + theta
        q = ((gamma-dp)+gamma) + dx
        r = p/q
        stpc = stp + r*(stx-stp)
        stpq = stp + (dp/(dp-dx))*(stx-stp)
        if abs(stpc-stp) > abs(stpq-stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True

    #     Third case: A lower function value, derivatives of the same sign,
    #     and the magnitude of the derivative decreases.
    elif abs(dp) < abs(dx):
        #        The cubic step is computed only if the cubic tends to infinity
        #        in the direction of the step or if the minimum of the cubic
        #        is beyond stp. Otherwise the cubic step is defined to be the
        #        secant step.
        theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        #        The case gamma = 0 only arises if the cubic does not tend
        #        to infinity in the direction of the step.
        gamma = s*sqrt(max(0,(theta/s)**2-(dx/s)*(dp/s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma-dp) + theta
        q = (gamma+(dx-dp)) + gamma
        r = p/q
        if r < 0 and gamma != 0:
            stpc = stp + r*(stx-stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin

        stpq = stp + (dp/(dp-dx))*(stx-stp)

        if brackt:
            #           A minimizer has been bracketed. If the cubic step is
            #           closer to stp than the secant step, the cubic step is
            #           taken, otherwise the secant step is taken.
            if abs(stpc-stp) < abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp+0.66*(sty-stp),stpf)
            else:
                stpf = max(stp+0.66*(sty-stp),stpf)
        else:
            #           A minimizer has not been bracketed. If the cubic step is
            #           farther from stp than the secant step, the cubic step is
            #           taken, otherwise the secant step is taken.

            if abs(stpc-stp) > abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax,stpf)
            stpf = max(stpmin,stpf)

    #     Fourth case: A lower function value, derivatives of the same sign,
    #     and the magnitude of the derivative does not decrease. If the
    #     minimum is not bracketed, the step is either stpmin or stpmax,
    #     otherwise the cubic step is taken.
    else:
        if brackt:
            theta = 3.0*(fp-fy)/(sty-stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))

            gamma = s*sqrt((theta/s)**2-(dy/s)*(dp/s))
            if stp > sty:  
                gamma = -gamma
            p = (gamma-dp) + theta
            q = ((gamma-dp)+gamma) + dy
            r = p/q
            stpc = stp + r*(sty-stp)
            stpf = stpc

        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx

        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax
