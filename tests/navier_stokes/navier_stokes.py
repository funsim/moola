""" This program optimises an control problem constrained by the Navier-Stokes equation """
from dolfin import *
from dolfin_adjoint import *
import moola
set_log_level(ERROR)

# Create a rectangle with a circular hole.
mesh = Mesh("mesh/mesh.xml")

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity
Q = FunctionSpace(mesh, "CG", 1)        # Pressure
D = FunctionSpace(mesh, "DG", 0)        # Control space
W = MixedFunctionSpace([V, Q])

# Define test and solution functions
v, q = TestFunctions(W)
s = Function(W)
u, p = split(s)

# Set parameter values
nu = Constant(1)     # Viscosity coefficient
f = Function(D)      # Control

# Define boundary conditions
noslip  = DirichletBC(W.sub(0), (0, 0), "on_boundary && x[0] > 0.0 && x[0] < 30")

inflow  = DirichletBC(W.sub(1), 1, "x[0] <= 0.0")
outflow = DirichletBC(W.sub(1), 0, "x[0] >= 30.0")

bcu = [noslip]
bcp = [inflow, outflow]

# Define the indicator function for the control area
chi = conditional(triangle.x[1] >= 5, 1, 0)

# Define the variational formulation of the Navier-Stokes equations
F = (inner(grad(u)*u, v)*dx +                 # Advection term
     nu*inner(grad(u), grad(v))*dx +          # Diffusion term
     inner(grad(p), v)*dx +                   # Pressure term
     inner(chi*f*u, v)*dx +                   # Sponge term
     div(u)*q*dx                              # Divergence term
    )

# Solve the Navier-Stokes equations
solve(F == 0, s, bcs=bcu+bcp)

# Define the optimisation proble,
alpha = Constant(1e-4)
J = Functional(inner(grad(u), grad(u))*dx + alpha*inner(f,f)*dx)
m = InitialConditionParameter(f)
rf = ReducedFunctional(J, m)

# Solve the optimisation problem
problem = rf.moola_problem()

solver_type = "moola"

###############  Moola solver #############################
if solver_type == "moola":
    #solver = moola.BFGS(tol=1e-200, options={'gtol': 1e-4, 'maxiter': 20, 'mem_lim': 20})
    solver = moola.NewtonCG(tol=1e-200, options={'gtol': 1e-4, 'maxiter': 20})

    m_moola = moola.DolfinPrimalVector(f)
    sol = solver.solve(problem, m_moola)

    m_opt = sol['Optimizer'].data

##############  Scipy solver ##############################
if solver_type == "scipy":
    m_opt = minimize(rf, method="Newton-CG", tol=1e-40, options={"maxiter": 10})

f.assign(m_opt)
solve(F == 0, s, bcs=bcu+bcp)
print "Functional value at optimizier: ", assemble(inner(grad(u), grad(u))*dx + alpha*inner(f,f)*dx)

plot(m_opt, interactive=True)
print moola.events
