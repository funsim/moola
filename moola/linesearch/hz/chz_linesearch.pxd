cdef extern from "cg_descent.c":

    ctypedef struct cg_parameter:
        double eps
        int PertRule
        int Wolfe
        long int nf
        long int ng 

    ctypedef struct cg_com:
        cg_parameter *Parm
        double eps
        int PertRule
        int Wolfe
        long int nf
        long int ng 
        double alpha         # stepsize along search direction
        double f             # function value for step alpha
        double f0            # old function value
        double df            # function derivative for step alpha
        double df0           # old function derivative value
        void *context        # user defined context

    void cg_printParms(cg_parameter *Parm)
    void cg_default(cg_parameter *Parm)
    int cg_line(cg_com *Com)

    int cg_Wolfe(double alpha, double f, double dphi, cg_com *Com)

