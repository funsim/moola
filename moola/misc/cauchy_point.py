''' Computes the Cauchy point as descirbed in Wright 1999, section 16.6 '''
import numpy as np
from .infinity import inf

def P(x, l, u):
    '''
    Projects x onto the bounds l and u:

                  / l_i if x_i < l_i,
    P(x, l, u) := | x_i if x_i in [l_i, u_i],
                  \ u_i if x_i > u_i.
    '''
    for i in range(len(x)):
        if x[i] < l[i]:
            x[i] = l[i] 
        elif x[i] > u[i]:
            x[i] = u[i]

def x(x0, g, t, l, u):
    ''' Returns P(x0 - g*t) '''
    x = x0.__class__(x0)
    x.axpy(-t, g)
    P(x, l, u)
    return x

def compute_cauchy_point(G, d, x0, l, u):
    ''' Computes the Cauchy point x_c := x0 - tg, 

    where t is the first local minimizer of a piecewise quadratic function q, i.e.:

    min_t q(x(t)) := 1/2x(t)^TGx(t) + x(t)^Td,

    with

    g := grad q(x0) = Gx0 + d,
    
    x(t) := P(x^0 - tg, l, u),

    and P the projection operator onto the bounds:

                  / l_i if x_i < l_i,
    P(x, l, u) := | x_i if x_i in [l_i, u_i],
                  \ u_i if x_i > u_i.

    The notation follows Wright 1999, section 16.6
    '''

    # Compute the gradient of q at x0 
    g = G(x0)
    g.axpy(1, d)

    # Compute values of t when kinks in x(t) occur
    tbar = g.__class__(g)
    for i in range(len(tbar)):
        if g[i] < 0 and u[i] < inf:
            tbar[i] = (x0[i] - u[i])/g[i]
        elif g[i] > 0 and l[i] > -inf:
            tbar[i] = (x0[i] - l[i])/g[i]
        else:
            tbar[i] = inf

    # Sort these t values in increasing order
    tbar_arr = tbar.array()
    sort_index = np.argsort(tbar_arr) # 1st element: index to smallest element, ... 
    t = lambda j: tbar_arr[sort_index[j]] if j >= 0 else 0. # A helper function to access the t values in order defined for -1 <= j < len(sort_index).

    # Loop over all the tbar's in order until we find the local minimum
    xc = None
    for j in range(len(sort_index)):
        # Update the search direction 
        # TODO Instead of recomputing p it could be updated 
        p = g.__class__(g)
        p.scale(-1)
        for i in range(len(p)):
            if not t(j-1) < tbar[i]:
                p[i] = 0 

        xt = x(x0, g, t(j-1), l, u)

        # Check if the minimizer lies within t(j-1) <= t < t(j)
        Gp = G(p)
        df = d.inner(p) + xt.inner(Gp)
        ddf = p.inner(Gp) 
        if ddf > 0 and 0 <= -df/ddf <= t(j) - t(j-1):
            # Local minimizer found
            t_opt = t(j-1) - df/ddf
            xc = x(x0, g, t_opt, l, u)
            break
        elif df > 0:
            # Local minimizer at t(j-1)
            xc = xt 
            break

    if not xc:
        # No mininizer was found in 0 <= t < t(len(sort_index)) 
        xc = x(x0, g, t(len(sort_index)-1), l, u)

    return xc
    

