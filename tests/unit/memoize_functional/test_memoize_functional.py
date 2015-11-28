from dolfin import *
from dolfin_adjoint import solve, Functional, Control, ReducedFunctional, adj_reset
import moola
import pytest
set_log_level(ERROR)

@pytest.fixture
def moola_problem():
    adj_reset()
    mesh = UnitSquareMesh(256, 256)
    V = FunctionSpace(mesh, "CG", 1)
    f = interpolate(Constant(1), V)
    u = Function(V)
    phi = TestFunction(V)
    F = inner(grad(u), grad(phi))*dx - f*phi*dx
    bc = DirichletBC(V, Constant(0), "on_boundary")
    solve(F == 0, u, bc)

    J = Functional(inner(u, u)*dx)
    m = Control(f)
    rf = ReducedFunctional(J, m)

    obj = rf.moola_problem().obj
    pf = moola.DolfinPrimalVector(f)

    return obj, pf

def test_IfRunningForwardModelTwice_ThenCachingTriggers(moola_problem):
    obj, pf = moola_problem
    timer = Timer("Eval")
    obj(pf)
    f1 = timer.stop()

    timer.start()
    obj(pf)
    f2 = timer.stop()

    assert f2/f1 < 1e-3

def test_IfRunningAdjointModel_ThenForwardEvaluationIsCached(moola_problem):
    obj, pf = moola_problem
    timer = Timer("Eval")
    obj.derivative(pf)
    f1 = timer.stop()

    timer.start()
    obj(pf)
    f2 = timer.stop()

    assert f2/f1 < 1e-3

def test_IfRunningForwardModel_ThenAdjointEvaluationIsPartiallyCached(moola_problem):
    obj, pf = moola_problem
    timer = Timer("Eval")
    obj(pf)
    f1 = timer.stop()

    timer.start()
    obj.derivative(pf)
    f2 = timer.stop()

    assert abs(f2/f1 - 1) < 0.3
