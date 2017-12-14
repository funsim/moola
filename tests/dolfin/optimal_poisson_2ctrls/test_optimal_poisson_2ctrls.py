""" Solves a MMS problem with smooth control """
from dolfin import *
from dolfin_adjoint import *
import moola

dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False

def solve_pde(u, V, f, g):
    v = TestFunction(V)
    F = (g*inner(grad(u), grad(v)) - f*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

def compute_errors(u, f, g):
    solve_pde(u, V, f, g)

    # Define the analytical expressions
    f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
    g_analytic = Expression("1")
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")

    # Compute the error
    control_error = errornorm(f_analytic, f) + errornorm(g_analytic, g)
    state_error = errornorm(u_analytic, u)

    print("Error in state: {}.".format(state_error))
    print("Error in control: {}.".format(control_error))

    return control_error, state_error


if __name__ == "__main__":

    n = 16
    mesh = UnitSquareMesh(n, n)

    def ref(mesh):
        cf = CellFunction("bool", mesh)
        subdomain = CompiledSubDomain('x[0]>.5')
        subdomain.mark(cf, True)
        return refine(mesh, cf)

    mesh = ref(mesh)
    mesh = ref(mesh)
    x = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "DG", 0)
    f = Function(W, name='Source')
    g = Function(W, name='Diffusivity')

    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1])

    alpha = Constant(1e-12)
    J = Functional(0.5*(inner(u-u_d, u-u_d))*dx + alpha*0.5*f**2*dx)

    # Run the forward model once to create the annotation
    solve_pde(u, V, f, g)

    # Run the optimisation
    m = SteadyParameter(f, value=f), SteadyParameter(g, value=g)
    m_moola = moola.DolfinPrimalVector((f, g))
    rf = ReducedFunctional(J, m)

    problem = rf.moola_problem()
    solver = moola.TrustRegionNewtonCG(problem, m_moola, options={'gtol': 1e-12, "tr_D0": 0.5})
    #solver = moola.NewtonCG(options={'gtol': 1e-10, 'maxiter': 20, 'ncg_reltol':1e-20, 'ncg_hesstol': "default"})
    #solver = moola.BFGS(tol=1e-200, options={'gtol': 1e-7, 'maxiter': 20, 'mem_lim': 20})
    sol = solver.solve()
    m_opt = sol['Optimizer'].data

    print(moola.events)
