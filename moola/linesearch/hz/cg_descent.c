/* =========================================================================
   ============================ CG_DESCENT =================================
   =========================================================================
       ________________________________________________________________
      |      A conjugate gradient method with guaranteed descent       |
      |             C-code Version 1.1  (October 6, 2005)              |
      |                    Version 1.2  (November 14, 2005)            |
      |                    Version 2.0  (September 23, 2007)           |
      |                    Version 3.0  (May 18, 2008)                 |
      |                    Version 4.0  (March 28, 2011)               |
      |                    Version 4.1  (April 8, 2011)                |
      |                    Version 4.2  (April 14, 2011)               |
      |                    Version 5.0  (May 1, 2011)                  |
      |                    Version 5.1  (January 31, 2012)             |
      |                    Version 5.2  (April 17, 2012)               |
      |                    Version 5.3  (May 18, 2012)                 |
      |                    Version 6.0  (November 6, 2012)             |
      |                    Version 6.1  (January 27, 2013)             |
      |                    Version 6.2  (February 2, 2013)             |
      |                    Version 6.3  (April 21, 2013)               |
      |                    Version 6.4  (April 29, 2013)               |
      |                    Version 6.5  (April 30, 2013)               |
      |                    Version 6.6  (May 28, 2013)                 |
      |                    Version 6.7  (April 7, 2014)                |
      |                                                                |
      |           William W. Hager    and   Hongchao Zhang             |
      |          hager@math.ufl.edu       hozhang@math.lsu.edu         |
      |                   Department of Mathematics                    |
      |                     University of Florida                      |
      |                 Gainesville, Florida 32611 USA                 |
      |                      352-392-0281 x 244                        |
      |                                                                |
      |                 Copyright by William W. Hager                  |
      |                                                                |
      |          http://www.math.ufl.edu/~hager/papers/CG              |
      |                                                                |
      |  Disclaimer: The views expressed are those of the authors and  |
      |              do not reflect the official policy or position of |
      |              the Department of Defense or the U.S. Government. |
      |                                                                |
      |      Approved for Public Release, Distribution Unlimited       |
      |________________________________________________________________|
       ________________________________________________________________
      |This program is free software; you can redistribute it and/or   |
      |modify it under the terms of the GNU General Public License as  |
      |published by the Free Software Foundation; either version 2 of  |
      |the License, or (at your option) any later version.             |
      |This program is distributed in the hope that it will be useful, |
      |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
      |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
      |GNU General Public License for more details.                    |
      |                                                                |
      |You should have received a copy of the GNU General Public       |
      |License along with this program; if not, write to the Free      |
      |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
      |MA  02110-1301  USA                                             |
      |________________________________________________________________|

      References:
      1. W. W. Hager and H. Zhang, A new conjugate gradient method
         with guaranteed descent and an efficient line search,
         SIAM Journal on Optimization, 16 (2005), 170-192.
      2. W. W. Hager and H. Zhang, Algorithm 851: CG_DESCENT,
         A conjugate gradient method with guaranteed descent,
         ACM Transactions on Mathematical Software, 32 (2006), 113-137.
      3. W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
         methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58.
      4. W. W. Hager and H. Zhang, Limited memory conjugate gradients,
         www.math.ufl.edu/~hager/papers/CG/lcg.pdf */

#include "cg_user.h"
#include "cg_descent.h"

/* begin external variables */
double one [1], zero [1] ;
/* end external variables */


/* =========================================================================
   ==== cg_Wolfe ===========================================================
   =========================================================================
   Check whether the Wolfe or the approximate Wolfe conditions are satisfied
   ========================================================================= */
int cg_Wolfe
(
    double   alpha, /* stepsize */
    double       f, /* function value associated with stepsize alpha */
    double    dphi, /* derivative value associated with stepsize alpha */
    cg_com    *Com  /* cg com */
)
{
    if ( dphi >= Com->wolfe_lo )
    {
        /* test original Wolfe conditions */
        if ( f - Com->f0 <= alpha*Com->wolfe_hi )
        {
            if ( Com->Parm->PrintLevel >= 2 )
            {
                printf ("Wolfe conditions hold\n") ;
/*              printf ("wolfe f: %25.15e f0: %25.15e df: %25.15e\n",
                         f, Com->f0, dphi) ;*/
            }
            return (1) ;
        }
        /* test approximate Wolfe conditions */
        else if ( Com->AWolfe )
        {
/*          if ( Com->Parm->PrintLevel >= 2 )
            {
                printf ("f:    %e fpert:    %e ", f, Com->fpert) ;
                if ( f > Com->fpert ) printf ("(fail)\n") ;
                else                  printf ("(OK)\n") ;
                printf ("dphi: %e hi bound: %e ", dphi, Com->awolfe_hi) ;
                if ( dphi > Com->awolfe_hi ) printf ("(fail)\n") ;
                else                         printf ("(OK)\n") ;
            }*/
            if ( (f <= Com->fpert) && (dphi <= Com->awolfe_hi) )
            {
                if ( Com->Parm->PrintLevel >= 2 )
                {
                    printf ("Approximate Wolfe conditions hold\n") ;
/*                  printf ("f: %25.15e fpert: %25.15e dphi: %25.15e awolf_hi: "
                           "%25.15e\n", f, Com->fpert, dphi, Com->awolfe_hi) ;*/
                }
                return (1) ;
            }
        }
    }
/*  else if ( Com->Parm->PrintLevel >= 2 )
    {
        printf ("dphi: %e lo bound: %e (fail)\n", dphi, Com->wolfe_lo) ;
    }*/
    return (0) ;
}


/* =========================================================================
   ==== cg_line ============================================================
   =========================================================================
   Approximate Wolfe line search routine
   Return:
      -2 (function nan)
       0 (Wolfe or approximate Wolfe conditions satisfied)
       3 (slope always negative in line search)
       4 (number line search iterations exceed nline)
       6 (excessive updating of eps)
       7 (Wolfe conditions never satisfied)
   ========================================================================= */
int cg_line
(
    cg_com   *Com /* cg com structure */
)
{
    int AWolfe, iter, ngrow, PrintLevel, qb, qb0, status, toggle ;
    double alpha, a, a1, a2, b, bmin, B, da, db, d0, d1, d2, dB, df, f, fa, fb,
           fB, a0, b0, da0, db0, fa0, fb0, width, rho ;
    char *s1, *s2, *fmt1, *fmt2 ;
    cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    if ( PrintLevel >= 1 )
    {
        if ( AWolfe )
        {
            printf ("Approximate Wolfe line search\n") ;
            printf ("=============================\n") ;
        }
        else
        {
            printf ("Wolfe line search\n") ;
            printf ("=================\n") ;
        }
    }

    /* evaluate function or gradient at Com->alpha (starting guess) */
    if ( Com->QuadOK )
    {
        status = cg_evaluate ("fg", "y", Com) ;
        fb = Com->f ;
        if ( !AWolfe ) fb -= Com->alpha*Com->wolfe_hi ;
        qb = TRUE ; /* function value at b known */
    }
    else
    {
        status = cg_evaluate ("g", "y", Com) ;
        qb = FALSE ;
    }
    if ( status ) return (status) ; /* function is undefined */
    b = Com->alpha ;

    if ( AWolfe )
    {
        db = Com->df ;
        d0 = da = Com->df0 ;
    }
    else
    {
        db = Com->df - Com->wolfe_hi ;
        d0 = da = Com->df0 - Com->wolfe_hi ;
    }
    a = ZERO ;
    a1 = ZERO ;
    d1 = d0 ;
    fa = Com->f0 ;
    if ( PrintLevel >= 1 )
    {
        fmt1 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb: %13.6e "
               "da: %13.6e db: %13.6e\n" ;
        fmt2 = "%9s %2s a: %13.6e b: %13.6e fa: %13.6e fb:  x.xxxxxxxxxx "
               "da: %13.6e db: %13.6e\n" ;
        if ( Com->QuadOK ) s2 = "OK" ;
        else               s2 = "" ;
        if ( qb ) printf (fmt1, "start    ", s2, a, b, fa, fb, da, db);
        else      printf (fmt2, "start    ", s2, a, b, fa, da, db) ;
    }

    /* if a quadratic interpolation step performed, check Wolfe conditions */
    if ( (Com->QuadOK) && (Com->f <= Com->f0) )
    {
        if ( cg_Wolfe (b, Com->f, Com->df, Com) ) return (0) ;
    }

    /* if a Wolfe line search and the Wolfe conditions have not been satisfied*/
    if ( !AWolfe ) Com->Wolfe = TRUE ;

    /*Find initial interval [a,b] such that
      da <= 0, db >= 0, fa <= fpert = [(f0 + eps*fabs (f0)) or (f0 + eps)] */
    rho = Com->rho ;
    ngrow = 1 ;
    while ( db < ZERO )
    {
        if ( !qb )
        {
            status = cg_evaluate ("f", "n", Com) ;
            if ( status ) return (status) ;
            if ( AWolfe ) fb = Com->f ;
            else          fb = Com->f - b*Com->wolfe_hi ;
            qb = TRUE ;
        }
        if ( fb > Com->fpert ) /* contract interval [a, b] */
        {
            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;   /* Wolfe conditions hold */
            if ( status == -2 ) goto Line ; /* db >= 0 */
            if ( Com->neps > Parm->neps ) return (6) ;
        }

        /* expansion phase */
        ngrow++ ;
        if ( ngrow > Parm->ntries ) return (3) ;
        /* update interval (a replaced by b) */
        a = b ;
        fa = fb ;
        da = db ;
        /* store old values of a and corresponding derivative */
        d2 = d1 ;
        d1 = da ;
        a2 = a1 ;
        a1 = a ;

        bmin = rho*b ;
        if ( (ngrow == 2) || (ngrow == 3) || (ngrow == 6) )
        {
            if ( d1 > d2 )
            {
                if ( ngrow == 2 )
                {
                    b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                }
                else
                {
                    if ( (d1-d2)/(a1-a2) >= (d2-d0)/a2 )
                    {
                        /* convex derivative, secant overestimates minimizer */
                        b = a1 - (a1-a2)*(d1/(d1-d2)) ;
                    }
                    else
                    {
                        /* concave derivative, secant underestimates minimizer*/
                        b = a1 - Parm->SecantAmp*(a1-a2)*(d1/(d1-d2)) ;
                    }
                }
                /* safeguard growth */
                b = MIN (b, Parm->ExpandSafe*a1) ;
            }
            else rho *= Parm->RhoGrow ;
        }
        else rho *= Parm->RhoGrow ;
        b = MAX (bmin, b) ;
        Com->alphaold = Com->alpha ;
        Com->alpha = b ;
        status = cg_evaluate ("g", "p", Com) ;
        if ( status ) return (status) ;
        b = Com->alpha ;
        qb = FALSE ;
        if ( AWolfe ) db = Com->df ;
        else          db = Com->df - Com->wolfe_hi ;
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            printf (fmt2, "expand   ", s2, a, b, fa, da, db) ;
        }
    }

    /* we now have fa <= fpert, da >= 0, db <= 0 */
Line:
    toggle = 0 ;
    width = b - a ;
    qb0 = FALSE ;
    for (iter = 0; iter < Parm->nline; iter++)
    {
        /* determine the next iterate */
        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {
            Com->QuadOK = TRUE ;
            if ( Com->UseCubic && qb )
            {
                s1 = "cubic    " ;
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
                if ( alpha < ZERO ) /* use secant method */
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else
            {
                s1 = "secant   " ;
                if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                else                 alpha = -1. ;
            }
            width = Parm->gamma*(b - a) ;
        }
        else if ( toggle == 1 ) /* iteration based on smallest value*/
        {
            Com->QuadOK = TRUE ;
            if ( Com->UseCubic )
            {
                s1 = "cubic    " ;
                if ( Com->alpha == a ) /* a is most recent iterate */
                {
                    alpha = cg_cubic (a0, fa0, da0, a, fa, da) ;
                }
                else if ( qb0 )        /* b is most recent iterate */
                {
                    alpha = cg_cubic (b, fb, db, b0, fb0, db0) ;
                }
                else alpha = -1. ;

                /* if alpha no good, use cubic between a and b */
                if ( (alpha <= a) || (alpha >= b) )
                {
                    if ( qb ) alpha = cg_cubic (a, fa, da, b, fb, db) ;
                    else alpha = -1. ;
                }

                /* if alpha still no good, use secant method */
                if ( alpha < ZERO )
                {
                    s1 = "secant   " ;
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
            else /* ( use secant ) */
            {
                s1 = "secant   " ;
                if ( (Com->alpha == a) && (da > da0) ) /* use a0 if possible */
                {
                    alpha = a - (a-a0)*(da/(da-da0)) ;
                }
                else if ( db < db0 )                   /* use b0 if possible */
                {
                    alpha = b - (b-b0)*(db/(db-db0)) ;
                }
                else /* secant based on a and b */
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }

                if ( (alpha <= a) || (alpha >= b) )
                {
                    if      ( -da < db ) alpha = a - (a-b)*(da/(da-db)) ;
                    else if ( da != db ) alpha = b - (a-b)*(db/(da-db)) ;
                    else                 alpha = -1. ;
                }
            }
        }
        else
        {
            alpha = .5*(a+b) ; /* use bisection if b-a decays slowly */
            s1 = "bisection" ;
            Com->QuadOK = FALSE ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            s1 = "bisection" ;
            if ( (alpha == a) || (alpha == b) ) return (7) ;
            Com->QuadOK = FALSE ; /* bisection was used */
        }

        if ( toggle == 0 ) /* save values for next iteration */
        {
            a0 = a ;
            b0 = b ;
            da0 = da ;
            db0 = db ;
            fa0 = fa ;
            if ( qb )
            {
                fb0 = fb ;
                qb0 = TRUE ;
            }
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        Com->alpha = alpha ;
        f = Com->f ;
        df = Com->df ;
        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) )
            {
                if ( PrintLevel >= 2 )
                {
                    printf ("             a: %13.6e f: %13.6e df: %13.6e %1s\n",
                             alpha, f, df, s1) ;
                }
                return (0) ;
            }
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= ZERO )
        {
            b = alpha ;
            fb = f ;
            db = df ;
            qb = TRUE ;
        }
        else if ( f <= Com->fpert )
        {
            a = alpha ;
            da = df ;
            fa = f ;
        }
        else
        {
            B = b ;
            if ( qb ) fB = fb ;
            dB = db ;
            b = alpha ;
            fb = f ;
            db = df ;
            /* contract interval [a, alpha] */
            status = cg_contract (&a, &fa, &da, &b, &fb, &db, Com) ;
            if ( status == 0 ) return (0) ;
            if ( status == -1 ) /* eps reduced, use [a, b] = [alpha, b] */
            {
                if ( Com->neps > Parm->neps ) return (6) ;
                a = b ;
                fa = fb ;
                da = db ;
                b = B ;
                if ( qb ) fb = fB ;
                db = dB ;
            }
            else qb = TRUE ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s2 = "OK" ;
            else               s2 = "" ;
            if ( !qb ) printf (fmt2, s1, s2, a, b, fa, da, db) ;
            else       printf (fmt1, s1, s2, a, b, fa, fb, da, db) ;
        }
    }
    return (4) ;
}

/* =========================================================================
   ==== cg_contract ========================================================
   =========================================================================
   The input for this routine is an interval [a, b] with the property that
   fa <= fpert, da >= 0, db >= 0, and fb >= fpert. The returned status is

  11  function or derivative not defined
   0  if the Wolfe conditions are satisfied
  -1  if a new value for eps is generated with the property that for the
      corresponding fpert, we have fb <= fpert
  -2  if a subinterval, also denoted [a, b], is generated with the property
      that fa <= fpert, da >= 0, and db <= 0

   NOTE: The input arguments are unchanged when status = -1
   ========================================================================= */
PRIVATE int cg_contract
(
    double    *A, /* left side of bracketing interval */
    double   *fA, /* function value at a */
    double   *dA, /* derivative at a */
    double    *B, /* right side of bracketing interval */
    double   *fB, /* function value at b */
    double   *dB, /* derivative at b */
    cg_com  *Com  /* cg com structure */
)
{
    int AWolfe, iter, PrintLevel, toggle, status ;
    double a, alpha, b, old, da, db, df, d1, dold, f, fa, fb, f1, fold,
           t, width ;
    char *s ;
    cg_parameter *Parm ;

    AWolfe = Com->AWolfe ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    a = *A ;
    fa = *fA ;
    da = *dA ;
    b = *B ;
    fb = *fB ;
    db = *dB ;
    f1 = fb ;
    d1 = db ;
    toggle = 0 ;
    width = ZERO ;
    for (iter = 0; iter < Parm->nshrink; iter++)
    {
        if ( (toggle == 0) || ((toggle == 2) && ((b-a) <= width)) )
        {
            /* cubic based on bracketing interval */
            alpha = cg_cubic (a, fa, da, b, fb, db) ;
            toggle = 0 ;
            width = Parm->gamma*(b-a) ;
            if ( iter ) Com->QuadOK = TRUE ; /* at least 2 cubic iterations */
        }
        else if ( toggle == 1 )
        {
            Com->QuadOK = TRUE ;
            /* cubic based on most recent iterate and smallest value */
            if ( old < a ) /* a is most recent iterate */
            {
                alpha = cg_cubic (a, fa, da, old, fold, dold) ;
            }
            else           /* b is most recent iterate */
            {
                alpha = cg_cubic (a, fa, da, b, fb, db) ;
            }
        }
        else
        {
            alpha = .5*(a+b) ; /* use bisection if b-a decays slowly */
            Com->QuadOK = FALSE ;
        }

        if ( (alpha <= a) || (alpha >= b) )
        {
            alpha = .5*(a+b) ;
            Com->QuadOK = FALSE ; /* bisection was used */
        }

        toggle++ ;
        if ( toggle > 2 ) toggle = 0 ;

        Com->alpha = alpha ;
        status = cg_evaluate ("fg", "n", Com) ;
        if ( status ) return (status) ;
        f = Com->f ;
        df = Com->df ;

        if ( Com->QuadOK )
        {
            if ( cg_Wolfe (alpha, f, df, Com) ) return (0) ;
        }
        if ( !AWolfe )
        {
            f -= alpha*Com->wolfe_hi ;
            df -= Com->wolfe_hi ;
        }
        if ( df >= ZERO )
        {
            *B = alpha ;
            *fB = f ;
            *dB = df ;
            *A = a ;
            *fA = fa ;
            *dA = da ;
            return (-2) ;
        }
        if ( f <= Com->fpert ) /* update a using alpha */
        {
            old = a ;
            a = alpha ;
            fold = fa ;
            fa = f ;
            dold = da ;
            da = df ;
        }
        else                     /* update b using alpha */
        {
            old = b ;
            b = alpha ;
            fb = f ;
            db = df ;
        }
        if ( PrintLevel >= 2 )
        {
            if ( Com->QuadOK ) s = "OK" ;
            else               s = "" ;
            printf ("contract  %2s a: %13.6e b: %13.6e fa: %13.6e fb: "
                    "%13.6e da: %13.6e db: %13.6e\n", s, a, b, fa, fb, da, db) ;
        }
    }

    /* see if the cost is small enough to change the PertRule */
    if ( fabs (fb) <= Com->SmallCost ) Com->PertRule = FALSE ;

    /* increase eps if slope is negative after Parm->nshrink iterations */
    t = Com->f0 ;
    if ( Com->PertRule )
    {
        if ( t != ZERO )
        {
            Com->eps = Parm->egrow*(f1-t)/fabs (t) ;
            Com->fpert = t + fabs (t)*Com->eps ;
        }
        else Com->fpert = 2.*f1 ;
    }
    else
    {
        Com->eps = Parm->egrow*(f1-t) ;
        Com->fpert = t + Com->eps ;
    }
    if ( PrintLevel >= 1 )
    {
        printf ("--increase eps: %e fpert: %e\n", Com->eps, Com->fpert) ;
    }
    Com->neps++ ;
    return (-1) ;
}

/* =========================================================================
   ==== cg_fg_evaluate =====================================================
   Evaluate the function and/or gradient.  Also, possibly check if either is nan
   and if so, then reduce the stepsize. Only used at the start of an iteration.
   Return:
      11 (function nan)
       0 (successful evaluation)
   =========================================================================*/

//PRIVATE int cg_evaluate
//(
//    char    *what, /* fg = evaluate func and grad, g = grad only,f = func only*/
//    char     *nan, /* y means check function/derivative values for nan */
//    cg_com   *Com
//)
//{
//    INT n ;
//    int i ;
//    double alpha, *d, *gtemp, *x, *xtemp ;
//    cg_parameter *Parm ;
//    Parm = Com->Parm ;
//    n = Com->n ;
//    x = Com->x ;
//    d = Com->d ;
//    xtemp = Com->xtemp ;
//    gtemp = Com->gtemp ;
//    alpha = Com->alpha ;
//    /* check to see if values are nan */
//    if ( !strcmp (nan, "y") || !strcmp (nan, "p") )
//    {
//        if ( !strcmp (what, "f") ) /* compute function */
//        {
//            cg_step (xtemp, x, d, alpha, n) ;
//            /* provisional function value */
//            Com->f = Com->cg_value (xtemp, n) ;
//            Com->nf++ ;
//
//            /* reduce stepsize if function value is nan */
//            if ( (Com->f != Com->f) || (Com->f >= INF) || (Com->f <= -INF) )
//            {
//                for (i = 0; i < Parm->ntries; i++)
//                {
//                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
//                    {
//                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
//                    }
//                    else                      /* multiply by nan_decay */
//                    {
//                        alpha *= Parm->nan_decay ;
//                    }
//                    cg_step (xtemp, x, d, alpha, n) ;
//                    Com->f = Com->cg_value (xtemp, n) ;
//                    Com->nf++ ;
//                    if ( (Com->f == Com->f) && (Com->f < INF) &&
//                         (Com->f > -INF) ) break ;
//                }
//                if ( i == Parm->ntries ) return (11) ;
//            }
//            Com->alpha = alpha ;
//        }
//        else if ( !strcmp (what, "g") ) /* compute gradient */
//        {
//            cg_step (xtemp, x, d, alpha, n) ;
//            Com->cg_grad (gtemp, xtemp, n) ;
//            Com->ng++ ;
//            Com->df = cg_dot (gtemp, d, n) ;
//            /* reduce stepsize if derivative is nan */
//            if ( (Com->df != Com->df) || (Com->df >= INF) || (Com->df <= -INF) )
//            {
//                for (i = 0; i < Parm->ntries; i++)
//                {
//                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
//                    {
//                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
//                    }
//                    else                      /* multiply by nan_decay */
//                    {
//                        alpha *= Parm->nan_decay ;
//                    }
//                    cg_step (xtemp, x, d, alpha, n) ;
//                    Com->cg_grad (gtemp, xtemp, n) ;
//                    Com->ng++ ;
//                    Com->df = cg_dot (gtemp, d, n) ;
//                    if ( (Com->df == Com->df) && (Com->df < INF) &&
//                         (Com->df > -INF) ) break ;
//                }
//                if ( i == Parm->ntries ) return (11) ;
//                Com->rho = Parm->nan_rho ;
//            }
//            else Com->rho = Parm->rho ;
//            Com->alpha = alpha ;
//        }
//        else                            /* compute function and gradient */
//        {
//            cg_step (xtemp, x, d, alpha, n) ;
//            if ( Com->cg_valgrad != NULL )
//            {
//                Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
//            }
//            else
//            {
//                Com->cg_grad (gtemp, xtemp, n) ;
//                Com->f = Com->cg_value (xtemp, n) ;
//            }
//            Com->df = cg_dot (gtemp, d, n) ;
//            Com->nf++ ;
//            Com->ng++ ;
//            /* reduce stepsize if function value or derivative is nan */
//            if ( (Com->df !=  Com->df) || (Com->f != Com->f) ||
//                 (Com->df >=  INF)     || (Com->f >= INF)    ||
//                 (Com->df <= -INF)     || (Com->f <= -INF))
//            {
//                for (i = 0; i < Parm->ntries; i++)
//                {
//                    if ( !strcmp (nan, "p") ) /* contract from good alpha */
//                    {
//                        alpha = Com->alphaold + .8*(alpha - Com->alphaold) ;
//                    }
//                    else                      /* multiply by nan_decay */
//                    {
//                        alpha *= Parm->nan_decay ;
//                    }
//                    cg_step (xtemp, x, d, alpha, n) ;
//                    if ( Com->cg_valgrad != NULL )
//                    {
//                        Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
//                    }
//                    else
//                    {
//                        Com->cg_grad (gtemp, xtemp, n) ;
//                        Com->f = Com->cg_value (xtemp, n) ;
//                    }
//                    Com->df = cg_dot (gtemp, d, n) ;
//                    Com->nf++ ;
//                    Com->ng++ ;
//                    if ( (Com->df == Com->df) && (Com->f == Com->f) &&
//                         (Com->df <  INF)     && (Com->f <  INF)    &&
//                         (Com->df > -INF)     && (Com->f > -INF) ) break ;
//                }
//                if ( i == Parm->ntries ) return (11) ;
//                Com->rho = Parm->nan_rho ;
//            }
//            else Com->rho = Parm->rho ;
//            Com->alpha = alpha ;
//        }
//    }
//    else                                /* evaluate without nan checking */
//    {
//        if ( !strcmp (what, "fg") )     /* compute function and gradient */
//        {
//            if ( alpha == ZERO )        /* evaluate at x */
//            {
//                /* the following copy is not needed except when the code
//                   is run using the MATLAB mex interface */
//                cg_copy (xtemp, x, n) ;
//                if ( Com->cg_valgrad != NULL )
//                {
//                    Com->f = Com->cg_valgrad (Com->g, xtemp, n) ;
//                }
//                else
//                {
//                    Com->cg_grad (Com->g, xtemp, n) ;
//                    Com->f = Com->cg_value (xtemp, n) ;
//                }
//            }
//            else
//            {
//                cg_step (xtemp, x, d, alpha, n) ;
//                if ( Com->cg_valgrad != NULL )
//                {
//                    Com->f = Com->cg_valgrad (gtemp, xtemp, n) ;
//                }
//                else
//                {
//                    Com->cg_grad (gtemp, xtemp, n) ;
//                    Com->f = Com->cg_value (xtemp, n) ;
//                }
//                Com->df = cg_dot (gtemp, d, n) ;
//            }
//            Com->nf++ ;
//            Com->ng++ ;
//            if ( (Com->df != Com->df) || (Com->f != Com->f) ||
//                 (Com->df == INF)     || (Com->f == INF)    ||
//                 (Com->df ==-INF)     || (Com->f ==-INF) ) return (11) ;
//        }
//        else if ( !strcmp (what, "f") ) /* compute function */
//        {
//            cg_step (xtemp, x, d, alpha, n) ;
//            Com->f = Com->cg_value (xtemp, n) ;
//            Com->nf++ ;
//            if ( (Com->f != Com->f) || (Com->f == INF) || (Com->f ==-INF) )
//                return (11) ;
//        }
//        else
//        {
//            cg_step (xtemp, x, d, alpha, n) ;
//            Com->cg_grad (gtemp, xtemp, n) ;
//            Com->df = cg_dot (gtemp, d, n) ;
//            Com->ng++ ;
//            if ( (Com->df != Com->df) || (Com->df == INF) || (Com->df ==-INF) )
//                return (11) ;
//        }
//    }
//    return (0) ;
//}

/* =========================================================================
   ==== cg_cubic ===========================================================
   =========================================================================
   Compute the minimizer of a Hermite cubic. If the computed minimizer
   outside [a, b], return -1 (it is assumed that a >= 0).
   ========================================================================= */
PRIVATE double cg_cubic
(
    double  a,
    double fa, /* function value at a */
    double da, /* derivative at a */
    double  b,
    double fb, /* function value at b */
    double db  /* derivative at b */
)
{
    double c, d1, d2, delta, t, v, w ;
    delta = b - a ;
    if ( delta == ZERO ) return (a) ;
    v = da + db - 3.*(fb-fa)/delta ;
    t = v*v - da*db ;
    if ( t < ZERO ) /* complex roots, use secant method */
    {
         if ( fabs (da) < fabs (db) ) c = a - (a-b)*(da/(da-db)) ;
         else if ( da != db )         c = b - (a-b)*(db/(da-db)) ;
         else                         c = -1 ;
         return (c) ;
    }

    if ( delta > ZERO ) w = sqrt(t) ;
    else                w =-sqrt(t) ;
    d1 = da + v - w ;
    d2 = db + v + w ;
    if ( (d1 == ZERO) && (d2 == ZERO) ) return (-1.) ;
    if ( fabs (d1) >= fabs (d2) ) c = a + delta*da/d1 ;
    else                          c = b - delta*db/d2 ;
    return (c) ;
}


/* =========================================================================
   === cg_default ==========================================================
   =========================================================================
   Set default conjugate gradient parameter values. If the parameter argument
   of cg_descent is NULL, this routine is called by cg_descent automatically.
   If the user wishes to set parameter values, then the cg_parameter structure
   should be allocated in the main program. The user could call cg_default
   to initialize the structure, and then individual elements in the structure
   could be changed, before passing the structure to cg_descent.
   =========================================================================*/
void cg_default
(
    cg_parameter   *Parm
)
{
    /* T => print final function value
       F => no printout of final function value */
    Parm->PrintFinal = FALSE ;

    /* Level 0 = no printing, ... , Level 3 = maximum printing */
    Parm->PrintLevel = 0 ;

    /* T => print parameters values
       F => do not display parameter values */
    Parm->PrintParms = FALSE ;

    /* T => use LBFGS
       F => only use L-BFGS when memory >= n */
    Parm->LBFGS = FALSE ;

    /* number of vectors stored in memory (code breaks in the Yk update if
       memory = 1 or 2) */
    Parm->memory = 11 ;

    /* SubCheck and SubSkip control the frequency with which the subspace
       condition is checked. It it checked for SubCheck*mem iterations and
       if it is not activated, then it is skipped for Subskip*mem iterations
       and Subskip is doubled. Whenever the subspace condition is satisfied,
       SubSkip is returned to its original value. */
    Parm->SubCheck = 8 ;
    Parm->SubSkip = 4 ;

    /* when relative distance from current gradient to subspace <= eta0,
       enter subspace if subspace dimension = mem (eta0 = 0 means gradient
       inside subspace) */
    Parm ->eta0 = 0.001 ; /* corresponds to eta0*eta0 in the paper */

    /* when relative distance from current gradient to subspace >= eta1,
       leave subspace (eta1 = 1 means gradient orthogonal to subspace) */
    Parm->eta1 = 0.900 ; /* corresponds to eta1*eta1 in the paper */

    /* when relative distance from current gradient to subspace <= eta2,
       always enter subspace (invariant space) */
    Parm->eta2 = 1.e-10 ;

    /* T => use approximate Wolfe line search
       F => use ordinary Wolfe line search, switch to approximate Wolfe when
                |f_k+1-f_k| < AWolfeFac*C_k, C_k = average size of cost */
    Parm->AWolfe = FALSE ;
    Parm->AWolfeFac = 1.e-3 ;

    /* factor in [0, 1] used to compute average cost magnitude C_k as follows:
       Q_k = 1 + (Qdecay)Q_k-1, Q_0 = 0,  C_k = C_k-1 + (|f_k| - C_k-1)/Q_k */
    Parm->Qdecay = .7 ;

    /* terminate after 2*n + nslow iterations without strict improvement in
       either function value or gradient */
    Parm->nslow = 1000 ;

    /* Stop Rules:
       T => ||grad||_infty <= max(grad_tol, initial |grad|_infty*StopFact)
       F => ||grad||_infty <= grad_tol*(1 + |f_k|) */
    Parm->StopRule = TRUE ;
    Parm->StopFac = 0.e-12 ;

    /* T => estimated error in function value is eps*Ck,
       F => estimated error in function value is eps */
    Parm->PertRule = TRUE ;
    Parm->eps = 1.e-6 ;

    /* factor by which eps grows when line search fails during contraction */
    Parm->egrow = 10. ;

    /* T => attempt quadratic interpolation in line search when
                |f_k+1 - f_k|/|f_k| > QuadCutOff
       F => no quadratic interpolation step */
    Parm->QuadStep = TRUE ;
    Parm->QuadCutOff = 1.e-12 ;

    /* maximum factor by which a quad step can reduce the step size */
    Parm->QuadSafe = 1.e-10 ;

    /* T => when possible, use a cubic step in the line search */
    Parm->UseCubic = TRUE ;

    /* use cubic step when |f_k+1 - f_k|/|f_k| > CubicCutOff */
    Parm->CubicCutOff = 1.e-12 ;

    /* |f| < SmallCost*starting cost => skip QuadStep and set PertRule = FALSE*/
    Parm->SmallCost = 1.e-30 ;

    /* T => check that f_k+1 - f_k <= debugtol*C_k
       F => no checking of function values */
    Parm->debug = FALSE ;
    Parm->debugtol = 1.e-10 ;

    /* if step is nonzero, it is the initial step of the initial line search */
    Parm->step = ZERO ;

    /* abort cg after maxit iterations */
    Parm->maxit = INT_INF ;

    /* maximum number of times the bracketing interval grows during expansion */
    Parm->ntries = (int) 50 ;

    /* maximum factor secant step increases stepsize in expansion phase */
    Parm->ExpandSafe = 200. ;

    /* factor by which secant step is amplified during expansion phase
       where minimizer is bracketed */
    Parm->SecantAmp = 1.05 ;

    /* factor by which rho grows during expansion phase where minimizer is
       bracketed */
    Parm->RhoGrow = 2.0 ;

    /* maximum number of times that eps is updated */
    Parm->neps = (int) 5 ;

    /* maximum number of times the bracketing interval shrinks */
    Parm->nshrink = (int) 10 ;

    /* maximum number of secant iterations in line search is nline */
    Parm->nline = (int) 50 ;

    /* conjugate gradient method restarts after (n*restart_fac) iterations */
    Parm->restart_fac = 6.0 ;

    /* stop when -alpha*dphi0 (estimated change in function value) <= feps*|f|*/
    Parm->feps = ZERO ;

    /* after encountering nan, growth factor when searching for
       a bracketing interval */
    Parm->nan_rho = 1.3 ;

    /* after encountering nan, decay factor for stepsize */
    Parm->nan_decay = 0.1 ;

    /* Wolfe line search parameter, range [0, .5]
       phi (a) - phi (0) <= delta phi'(0) */
    Parm->delta = .1 ;

    /* Wolfe line search parameter, range [delta, 1]
       phi' (a) >= sigma phi' (0) */
    Parm->sigma = .9 ;

    /* decay factor for bracket interval width in line search, range (0, 1) */
    Parm->gamma = .66 ;

    /* growth factor in search for initial bracket interval */
    Parm->rho = 5. ;

    /* starting guess for line search =
         psi0 ||x_0||_infty over ||g_0||_infty if x_0 != 0
         psi0 |f(x_0)|/||g_0||_2               otherwise */
    Parm->psi0 = .01 ;      /* factor used in starting guess for iteration 1 */

    /* for a QuadStep, function evaluated on interval
       [psi_lo, phi_hi]*psi2*previous step */
    Parm->psi_lo = 0.1 ;
    Parm->psi_hi = 10. ;

    /* when the function is approximately quadratic, use gradient at
       psi1*psi2*previous step for estimating initial stepsize */
    Parm->psi1 = 1.0 ;

    /* when starting a new cg iteration, our initial guess for the line
       search stepsize is psi2*previous step */
    Parm->psi2 = 2. ;

    /* choose theta adaptively if AdaptiveBeta = T */
    Parm->AdaptiveBeta = FALSE ;

    /* lower bound for beta is BetaLower*d_k'g_k/ ||d_k||^2 */
    Parm->BetaLower = 0.4 ;

    /* value of the parameter theta in the cg_descent update formula:
       W. W. Hager and H. Zhang, A survey of nonlinear conjugate gradient
       methods, Pacific Journal of Optimization, 2 (2006), pp. 35-58. */
    Parm->theta = 1.0 ;

    /* parameter used in cost error estimate for quadratic restart criterion */
    Parm->qeps = 1.e-12 ;

    /* number of iterations the function is nearly quadratic before a restart */
    Parm->qrestart = 6 ;

    /* treat cost as quadratic if
       |1 - (cost change)/(quadratic cost change)| <= qrule */
    Parm->qrule = 1.e-8 ;
}

/* =========================================================================
   ==== cg_printParms ======================================================
   =========================================================================
   Print the contents of the cg_parameter structure
   ========================================================================= */
void cg_printParms
(
    cg_parameter  *Parm
)
{
    printf ("PARAMETERS:\n") ;
    printf ("\n") ;
    printf ("Wolfe line search parameter ..................... delta: %e\n",
             Parm->delta) ;
    printf ("Wolfe line search parameter ..................... sigma: %e\n",
             Parm->sigma) ;
    printf ("decay factor for bracketing interval ............ gamma: %e\n",
             Parm->gamma) ;
    printf ("growth factor for bracket interval ................ rho: %e\n",
             Parm->rho) ;
    printf ("growth factor for bracket interval after nan .. nan_rho: %e\n",
             Parm->nan_rho) ;
    printf ("decay factor for stepsize after nan ......... nan_decay: %e\n",
             Parm->nan_decay) ;
    printf ("parameter in lower bound for beta ........... BetaLower: %e\n",
             Parm->BetaLower) ;
    printf ("parameter describing cg_descent family .......... theta: %e\n",
             Parm->theta) ;
    printf ("perturbation parameter for function value ......... eps: %e\n",
             Parm->eps) ;
    printf ("factor by which eps grows if necessary .......... egrow: %e\n",
             Parm->egrow) ;
    printf ("factor for computing average cost .............. Qdecay: %e\n",
             Parm->Qdecay) ;
    printf ("relative change in cost to stop quadstep ... QuadCutOff: %e\n",
             Parm->QuadCutOff) ;
    printf ("maximum factor quadstep reduces stepsize ..... QuadSafe: %e\n",
             Parm->QuadSafe) ;
    printf ("skip quadstep if |f| <= SmallCost*start cost  SmallCost: %e\n",
             Parm->SmallCost) ;
    printf ("relative change in cost to stop cubic step  CubicCutOff: %e\n",
             Parm->CubicCutOff) ;
    printf ("terminate if no improvement over nslow iter ..... nslow: %i\n",
             Parm->nslow) ;
    printf ("factor multiplying gradient in stop condition . StopFac: %e\n",
             Parm->StopFac) ;
    printf ("cost change factor, approx Wolfe transition . AWolfeFac: %e\n",
             Parm->AWolfeFac) ;
    printf ("restart cg every restart_fac*n iterations . restart_fac: %e\n",
             Parm->restart_fac) ;
    printf ("cost error in quadratic restart is qeps*cost ..... qeps: %e\n",
             Parm->qeps) ;
    printf ("number of quadratic iterations before restart  qrestart: %i\n",
             Parm->qrestart) ;
    printf ("parameter used to decide if cost is quadratic ... qrule: %e\n",
             Parm->qrule) ;
    printf ("stop when cost change <= feps*|f| ................ feps: %e\n",
             Parm->feps) ;
    printf ("starting guess parameter in first iteration ...... psi0: %e\n",
             Parm->psi0) ;
    printf ("starting step in first iteration if nonzero ...... step: %e\n",
             Parm->step) ;
    printf ("lower bound factor in quad step ................ psi_lo: %e\n",
             Parm->psi_lo) ;
    printf ("upper bound factor in quad step ................ psi_hi: %e\n",
             Parm->psi_hi) ;
    printf ("initial guess factor for quadratic functions ..... psi1: %e\n",
             Parm->psi1) ;
    printf ("initial guess factor for general iteration ....... psi2: %e\n",
             Parm->psi2) ;
    printf ("max iterations .................................. maxit: %i\n",
             (int) Parm->maxit) ;
    printf ("max number of contracts in the line search .... nshrink: %i\n",
             Parm->nshrink) ;
    printf ("max expansions in line search .................. ntries: %i\n",
             Parm->ntries) ;
    printf ("maximum growth of secant step in expansion . ExpandSafe: %e\n",
             Parm->ExpandSafe) ;
    printf ("growth factor for secant step during expand . SecantAmp: %e\n",
             Parm->SecantAmp) ;
    printf ("growth factor for rho during expansion phase .. RhoGrow: %e\n",
             Parm->RhoGrow) ;
    printf ("distance threshhold for entering subspace ........ eta0: %e\n",
             Parm->eta0) ;
    printf ("distance threshhold for leaving subspace ......... eta1: %e\n",
             Parm->eta1) ;
    printf ("distance threshhold for invariant space .......... eta2: %e\n",
             Parm->eta2) ;
    printf ("number of vectors stored in memory ............. memory: %i\n",
             Parm->memory) ;
    printf ("check subspace condition mem*SubCheck its .... SubCheck: %i\n",
             Parm->SubCheck) ;
    printf ("skip subspace checking for mem*SubSkip its .... SubSkip: %i\n",
             Parm->SubSkip) ;
    printf ("max number of times that eps is updated .......... neps: %i\n",
             Parm->neps) ;
    printf ("max number of iterations in line search ......... nline: %i\n",
             Parm->nline) ;
    printf ("print level (0 = none, 3 = maximum) ........ PrintLevel: %i\n",
             Parm->PrintLevel) ;
    printf ("Logical parameters:\n") ;
    if ( Parm->PertRule )
        printf ("    Error estimate for function value is eps*Ck\n") ;
    else
        printf ("    Error estimate for function value is eps\n") ;
    if ( Parm->QuadStep )
        printf ("    Use quadratic interpolation step\n") ;
    else
        printf ("    No quadratic interpolation step\n") ;
    if ( Parm->UseCubic)
        printf ("    Use cubic interpolation step when possible\n") ;
    else
        printf ("    Avoid cubic interpolation steps\n") ;
    if ( Parm->AdaptiveBeta )
        printf ("    Adaptively adjust direction update parameter beta\n") ;
    else
        printf ("    Use fixed parameter theta in direction update\n") ;
    if ( Parm->PrintFinal )
        printf ("    Print final cost and statistics\n") ;
    else
        printf ("    Do not print final cost and statistics\n") ;
    if ( Parm->PrintParms )
        printf ("    Print the parameter structure\n") ;
    else
        printf ("    Do not print parameter structure\n") ;
    if ( Parm->AWolfe)
        printf ("    Approximate Wolfe line search\n") ;
    else
        printf ("    Wolfe line search") ;
        if ( Parm->AWolfeFac > 0. )
            printf (" ... switching to approximate Wolfe\n") ;
        else
            printf ("\n") ;
    if ( Parm->StopRule )
        printf ("    Stopping condition uses initial grad tolerance\n") ;
    else
        printf ("    Stopping condition weighted by absolute cost\n") ;
    if ( Parm->debug)
        printf ("    Check for decay of cost, debugger is on\n") ;
    else
        printf ("    Do not check for decay of cost, debugger is off\n") ;
}

/*
Version 1.2 Change:
  1. The variable dpsi needs to be included in the argument list for
     subroutine cg_updateW (update of a Wolfe line search)

Version 2.0 Changes:
     The user interface was redesigned. The parameters no longer need to
     be read from a file. For compatibility with earlier versions of the
     code, we include the routine cg_readParms to read parameters.
     In the simplest case, the user can use NULL for the
     parameter argument of cg_descent, and the code sets the default
     parameter values. If the user wishes to modify the parameters, call
     cg_default in the main program to initialize a cg_parameter
     structure. Individual elements of the structure could be modified.
     The header file cg_user.h contains the structures and prototypes
     that the user may need to reference or modify, while cg_descent.h
     contains header elements that only cg_descent will access.  Note
     that the arguments of cg_descent have changed.

Version 3.0 Changes:
     Major overhaul

Version 4.0 Changes:
  Modifications 1-3 were made based on results obtained by Yu-Hong Dai and
  Cai-Xia Kou in the paper "A nonlinear conjugate gradient algorithm with an
     optimal property and an improved Wolfe line search"
  1. Set theta = 1.0 by default in the cg_descent rule for beta_k (Dai and
     Kou showed both theoretical and practical advantages in this choice).
  2. Increase the default value of restart_fac to 6 (a value larger than 1 is
     more efficient when the problem dimension is small)
  3. Restart the CG iteration if the objective function is nearly quadratic
     for several iterations (qrestart).
  4. New lower bound for beta: BetaLower*d_k'g_k/ ||d_k||^2. This lower
     bound guarantees convergence and it seems to provide better
     performance than the original lower bound for beta in cg_descent;
     it also seems to give slightly better performance than the
     lower bound BetaLower*d_k'g_k+1/ ||d_k||^2 suggested by Dai and Kou.
  5. Evaluation of the objective function and gradient is now handled by
     the routine cg_evaluate.

Version 4.1 Changes:
  1. Change cg_tol to be consistent with corresponding routine in asa_cg
  2. Compute dnorm2 when d is evaluated and make loops consistent with asa_cg

Version 4.2 Changes:
  1. Modify the line search so that when there are too many contractions,
     the code will increase eps and switch to expansion of the search interval.
     This fixes some cases where the code terminates when eps is too small.
     When the estimated error in the cost function is too small, the algorithm
     could fail in cases where the slope is negative at both ends of the
     search interval and the objective function value on the right side of the
     interval is larger than the value at the left side (because the true
     objective function value on the right is not greater than the value on
     the left).
  2. Fix bug in cg_lineW

Version 5.0 Changes:
     Revise the line search routines to exploit steps based on the
     minimizer of a Hermite interpolating cubic. Combine the approximate
     and the ordinary Wolfe line search into a single routine.
     Include safeguarded extrapolation during the expansion phase
     of the line search. Employ a quadratic interpolation step even
     when the requirement ftemp < f for a quadstep is not satisfied.

Version 5.1 Changes:
  1. Shintaro Kaneko pointed out spelling error in line 738, change
     "stict" to "strict"
  2. Add MATLAB interface

Version 5.2 Changes:
  1. Make QuadOK always TRUE in the quadratic interpolation step.
  2. Change QuadSafe to 1.e-10 instead of 1.e-3 (less safe guarding).
  3. Change psi0 to 2 in the routine for computing the stepsize
     in the first iteration when x = 0.
  4. In the quadratic step routine, the stepsize at the trial point
     is the maximum of {psi_lo*psi2, prior df/(current df)}*previous step
  5. In quadratic step routine, the function is evaluated on the safe-guarded
     interval [psi_lo, phi_hi]*psi2*previous step.
  6. Allow more rapid expansion in the line search routine.

Version 5.3 Changes:
  1. Make changes so that cg_descent works with version R2012a of MATLAB.
     This required allocating memory for the work array inside the MATLAB
     mex routine and rearranging memory so that the xtemp pointer and the
     work array pointer are the same.

Version 6.0 Changes:
  Major revision of the code to implement the limited memory conjugate
  gradient algorithm documented in reference [4] at the top of this file.
  The code has doubled in length. The line search remains the same except
  for small adjustments in the nan detection which allows it to
  solve more problems that generate nan's. The direction routine, however,
  is completely new. Version 5.3 of the code is obtained by setting
  the memory parameter to be 0. The L-BFGS algorithm is obtained by setting
  the LBFGS parameter to TRUE. Otherwise, for memory > 0 and LBFGS = FALSE,
  the search directions are obtained by a limited memory version of the
  original cg_descent algorithm. The memory is used to detect when the
  gradients lose orthogonality locally. When orthogonality is lost,
  the algorithm solves a subspace problem until orthogonality is restored.
  It is now possible to utilize the BLAS, if they are available, by
  commenting out a line in the file cg_blas.h. See the README file for
  the details.

Version 6.1 Changes:
  Fixed problems connected with memory handling in the MATLAB version
  of the code. These errors only arise when using some versions of MATLAB.
  Replaced "malloc" in the cg_descent mex routine with "mxMalloc".
  Thanks to Stephen Vavasis for reporting this error that occurred when
  using MATLAB version R2012a, 7.14.

Version 6.2 Change:
  When using cg_descent in MATLAB, the input starting guess is no longer
  overwritten by the final solution. This makes the cg_descent mex function
  compliant with MATLAB's convention for the treatment of input arguments.
  Thanks to Dan Scholnik, Naval Research Laboratory, for pointing out this
  inconsistency with MATLAB convention in earlier versions of cg_descent.

Version 6.3 Change:
  For problems of dimension <= Parm->memory (default 11), the final
  solution was not copied to the user's solution argument x. Instead x
  stored the the next-to-final iterate. This final copy is now inserted
  on line 969.  Thanks to Arnold Neumaier of the University of Vienna
  for pointing out this bug.

Version 6.4 Change:
  In order to prevent a segmentation fault connected with MATLAB's memory
  handling, inserted a copy statement inside cg_evaluate. This copy is
  only needed when solving certain problems with MATLAB.  Many thanks
  to Arnold Neumaier for pointing out this problem

Version 6.5 Change:
  When using the code in MATLAB, changed the x vector that is passed to
  the user's function and gradient routine to be column vectors instead
  of row vectors.

Version 6.6 Change:
  Expand the statistics structure to include the number of subspaces (NumSub)
  and the number of subspace iterations (IterSub).

Version 6.7 Change:
  Add interface to CUTEst
*/
