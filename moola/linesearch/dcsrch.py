
def dcsrch(stp, f, g, ftol, gtol, xtol, task, stpmin, stpmax, isave, dsave):
    """
     Subroutine dcsrch

     This subroutine finds a step that satisfies a sufficient
     decrease condition and a curvature condition.

     Each call of the subroutine updates an interval with
     endpoints stx and sty. The interval is initially chosen
     so that it contains a minimizer of the modified function

           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

     If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
     interval is chosen so that it contains a minimizer of f.

     The algorithm is designed to find a step that satisfies
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

     A typical invocation of dcsrch has the following outline:

     Evaluate the function at stp = 0.0d0; store in f.
     Evaluate the gradient at stp = 0.0d0; store in g.
     Choose a starting step stp.

     task = 'START'
  10 continue
        call dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
    +               isave,dsave)
        if (task == 'FG') then
           Evaluate the function and the gradient at stp
           go to 10
           end if

     NOTE: The user must not alter work arrays between calls.

     The subroutine statement is

       subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
                         task,isave,dsave)
     where

       stp is a double precision variable.
         On entry stp is the current estimate of a satisfactory
            step. On initial entry, a positive initial estimate
            must be provided.
         On exit stp is the current estimate of a satisfactory step
            if task = 'FG'. If task = 'CONV' then stp satisfies
            the sufficient decrease and curvature condition.

       f is a double precision variable.
         On initial entry f is the value of the function at 0.
            On subsequent entries f is the value of the
            function at stp.
         On exit f is the value of the function at stp.

       g is a double precision variable.
         On initial entry g is the derivative of the function at 0.
            On subsequent entries g is the derivative of the
            function at stp.
         On exit g is the derivative of the function at stp.

       ftol is a double precision variable.
         On entry ftol specifies a nonnegative tolerance for the
            sufficient decrease condition.
         On exit ftol is unchanged.

       gtol is a double precision variable.
         On entry gtol specifies a nonnegative tolerance for the
            curvature condition.
         On exit gtol is unchanged.

       xtol is a double precision variable.
         On entry xtol specifies a nonnegative relative tolerance
            for an acceptable step. The subroutine exits with a
            warning if the relative difference between sty and stx
            is less than xtol.
         On exit xtol is unchanged.

       task is a character variable of length at least 60.
         On initial entry task must be set to 'START'.
         On exit task indicates the required action:

            If task(1:2) = 'FG' then evaluate the function and
            derivative at stp and call dcsrch again.

            If task(1:4) = 'CONV' then the search is successful.

            If task(1:4) = 'WARN' then the subroutine is not able
            to satisfy the convergence conditions. The exit value of
            stp contains the best point found during the search.

            If task(1:5) = 'ERROR' then there is an error in the
            input arguments.

         On exit with convergence, a warning or an error, the
            variable task contains additional information.

       stpmin is a double precision variable.
         On entry stpmin is a nonnegative lower bound for the step.
         On exit stpmin is unchanged.

       stpmax is a double precision variable.
         On entry stpmax is a nonnegative upper bound for the step.
         On exit stpmax is unchanged.

       isave is an integer work array of dimension 2.

       dsave is a double precision work array of dimension 13.

     Subprograms called

       MINPACK-2 ... dcstep

     MINPACK-1 Project. June 1983.
     Argonne National Laboratory.
     Jorge J. More' and David J. Thuente.

     MINPACK-2 Project. November 1993.
     Argonne National Laboratory and University of Minnesota.
     Brett M. Averick, Richard G. Carter, and Jorge J. More'.

     Converted and extended to Python, July 2013
     Imperial College London
     Simon W. Funke
    """

    from .dcstep import dcstep

    xtrapl=1.1
    xtrapu=4.0

    # Initialization block.
    if task == 'START':

        if stpmin is None:
          stpmin = 0
        if stpmax is None:
          stpmax = 1e999

        # Check the input arguments for errors.
        if stp < stpmin: 
            task = 'Error: stp < stpmin'
        if stp > stpmax: 
            task = 'Error: stp > stpmax'
        if g >= 0.0: 
            task = 'Error: initial g >= 0 (%e)' % g
        if ftol < 0.0: 
            task = 'Error: ftol < zero'
        if gtol < 0.0:  
            task = 'Error: gtol < zero'
        if xtol < 0.0:  
            task = 'Error: xtol < zero'
        if stpmin < 0.0: 
            task = 'Error: stpmin < zero'
        if stpmax < stpmin: 
            task = 'Error: stpmax < stpmin'

        # Exit if there are errors on input.
        if "Error" in task:
            return stp, task, isave, dsave

        # Initialize local variables.
        brackt = False
        stage = 1
        finit = f
        ginit = g
        gtest = ftol*ginit
        width = stpmax - stpmin
        width1 = width/0.5

        # The variables stx, fx, gx contain the values of the step,
        # function, and derivative at the best step.
        # The variables sty, fy, gy contain the value of the step,
        # function, and derivative at sty.
        # The variables stp, f, g contain the values of the step,
        # function, and derivative at stp.

        stx = 0.0
        fx = finit
        gx = ginit
        sty = 0.0
        fy = finit
        gy = ginit
        stmin = 0.0
        stmax = stp + xtrapu*stp
        task = 'FG'

        # Save local variables.
        isave[1] = brackt
        isave[2] = stage
        dsave[1] = ginit
        dsave[2] = gtest
        dsave[3] = gx
        dsave[4] = gy
        dsave[5] = finit
        dsave[6] = fx
        dsave[7] = fy
        dsave[8] = stx
        dsave[9] = sty
        dsave[10] = stmin
        dsave[11] = stmax
        dsave[12] = width
        dsave[13] = width1

        return stp, task, isave, dsave

    else:
        # Restore local variables.
        brackt = isave[1]
        stage = isave[2]
        ginit = dsave[1]
        gtest = dsave[2]
        gx = dsave[3]
        gy = dsave[4]
        finit = dsave[5]
        fx = dsave[6]
        fy = dsave[7]
        stx = dsave[8]
        sty = dsave[9]
        stmin = dsave[10]
        stmax = dsave[11]
        width = dsave[12]
        width1 = dsave[13]

    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    # algorithm enters the second stage.
    ftest = finit + stp*gtest
    if stage == 1 and f <= ftest and g >= 0.0: 
        stage = 2

    # Test for warnings.
    if brackt and (stp <= stmin or stp >= stmax): 
        task = 'Warning: Rounding errors prevent progress'
    if brackt and stmax-stmin <= xtol*stmax:
        task = 'Warning: xtol test satisfied'
    if stp == stpmax and f <= ftest and g <= gtest:
        task = 'Warning: stp = stpmax'
    if stp == stpmin and (f > ftest or g >= gtest):
        task = 'Warning: stp = stpmin'

    # Test for convergence.
    if f <= ftest and abs(g) <= gtol*(-ginit):
        task = 'Convergence'

    # Test for termination.
    if "Warning" in task or task == 'Convergence': 
        # Save local variables.
        isave[1] = brackt
        isave[2] = stage
        dsave[1] = ginit
        dsave[2] = gtest
        dsave[3] = gx
        dsave[4] = gy
        dsave[5] = finit
        dsave[6] = fx
        dsave[7] = fy
        dsave[8] = stx
        dsave[9] = sty
        dsave[10] = stmin
        dsave[11] = stmax
        dsave[12] = width
        dsave[13] = width1
        return stp, task, isave, dsave

    # A modified function is used to predict the step during the
    # first stage if a lower function value has been obtained but
    # the decrease is not sufficient.
    if stage == 1 and f <= fx and f > ftest:
        # Define the modified function and derivative values.
        fm = f - stp*gtest
        fxm = fx - stx*gtest
        fym = fy - sty*gtest
        gm = g - gtest
        gxm = gx - gtest
        gym = gy - gtest

        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax = dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

        # Reset the function and derivative values for f.
        fx = fxm + stx*gtest
        fy = fym + sty*gtest
        gx = gxm + gtest
        gy = gym + gtest

    else:
        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax = dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)

    # Decide if a bisection step is needed.
    if brackt:
        if abs(sty-stx) >= 0.66*width1: 
            stp = stx + 0.5*(sty-stx)
        width1 = width
        width = abs(sty-stx)

    # Set the minimum and maximum steps allowed for stp.
    if brackt:
        stmin = min(stx,sty)
        stmax = max(stx,sty)
    else:
        stmin = stp + xtrapl*(stp-stx)
        stmax = stp + xtrapu*(stp-stx)

    # Force the step to be within the bounds stpmax and stpmin.
    stp = max(stp, stpmin)
    stp = min(stp, stpmax)

    # If further progress is not possible, let stp be the best
    # point obtained during the search.
    if brackt and (stp <= stmin or stp >= stmax) or (brackt and stmax-stmin <= xtol*stmax): 
        stp = stx

    # Obtain another function and derivative.
    task = 'FG'

    # Save local variables.
    isave[1] = brackt
    isave[2] = stage
    dsave[1] = ginit
    dsave[2] = gtest
    dsave[3] = gx
    dsave[4] = gy
    dsave[5] = finit
    dsave[6] = fx
    dsave[7] = fy
    dsave[8] = stx
    dsave[9] = sty
    dsave[10] = stmin
    dsave[11] = stmax
    dsave[12] = width
    dsave[13] = width1
    return stp, task, isave, dsave
