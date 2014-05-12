cimport chz_linesearch as chz

cdef public int cg_evaluate(char *what, char *nan, chz.cg_com *com):
    phi, phi_dphi = <object> com.context
    alpha = com.alpha

    if  what == b"g":
        print "Evaluating g"
        print "for alpha: ", com.alpha
        com.df = phi_dphi(alpha)[1]
    else:
        print "What ", what
        raise ValueError, "Unkown value for parameter what"

    return 0


def phi(alpha):
    return 10*alpha**2 + 2*alpha + 10

def phi_dphi(alpha):
    return phi(alpha), 2*10*alpha + 2

cdef class HZLineSearch:
    cdef chz.cg_parameter c_params
    cdef chz.cg_com c_com

    def __cinit__(self):
        chz.cg_default(&self.c_params)

        self.c_com.Parm = &(self.c_params)
        self.c_com.eps = self.c_params.eps
        self.c_com.PertRule = self.c_params.PertRule
        self.c_com.nf = 0
        self.c_com.ng = 0

    def print_parameter(self):
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

        status = chz.cg_line(&(self.c_com))
        if status != 0:
            print "Warning: Line search failed. Reason: %s" % msgs[status]

    def cg_Wolfe(self, alpha, f, dphi):
        chz.cg_Wolfe(alpha, f, dphi, &(self.c_com))
