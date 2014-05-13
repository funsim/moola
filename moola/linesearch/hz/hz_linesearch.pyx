cimport chz_linesearch as chz

cdef public int cg_evaluate(char *what, char *nan, chz.cg_com *com):
    phi, phi_dphi = <object> com.context
    alpha = com.alpha

    if  what == b"g":
        print "Evaluating g"
        print "for alpha: ", com.alpha
        com.df = phi_dphi(alpha)[1]
    elif  what == b"f":
        print "Evaluating f"
        print "for alpha: ", com.alpha
        com.f = phi(alpha)
    elif  what == b"fg":
        print "Evaluating f and g"
        print "for alpha: ", com.alpha
        com.f, com.df = phi_dphi(alpha)
    else:
        print "What ", what
        raise ValueError, "Unkown value for parameter what"

    print "Returning f =", com.f, " and g = ", com.df
    return 0


def phi(alpha):
    x0 = 2
    return 0.5*(alpha - x0)**2

def phi_dphi(alpha):
    x0 = 2
    return phi(alpha), alpha - x0

cdef class HZLineSearch:
    cdef chz.cg_parameter c_params
    cdef chz.cg_com c_com

    def __cinit__(self):
        chz.cg_default(&self.c_params)

        self.c_com.Parm = &(self.c_params)
        self.c_com.eps = self.c_params.eps
        self.c_com.PertRule = self.c_params.PertRule

    def set_print_level(self, level):
        ''' Level 0  = no printing) 
            ...
            Level 3 = maximum printing 
        '''
        self.c_params.PrintLevel = level

    def print_parameters(self):
        chz.cg_printParms(&self.c_params)

    def search(self, phi=phi, phi_dphi=phi_dphi):
        msgs = {
                -2: "function nan",
                 0: "Wolfe or approximate Wolfe conditions satisfied",
                 3: "slope always negative in line search",
                 4: "number line search iterations exceed nline",
                 6: "excessive updating of eps",
                 7: "Wolfe conditions never satisfied",
                }

        context = (phi, phi_dphi)
        self.c_com.context = <void*> context

        self.c_com.f0 = phi(0)
        self.c_com.df0 = phi_dphi(0)[1]

        status = chz.cg_line(&(self.c_com))
        if status != 0:
            print "Warning: Line search failed. Reason: %s" % msgs[status]

    def cg_Wolfe(self, alpha, f, dphi):
        chz.cg_Wolfe(alpha, f, dphi, &(self.c_com))
