from math import sqrt
from moola import *
from dolfin import *
import pytest

@pytest.fixture
def primal():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)

    f = interpolate(Constant(2), V)
    g = interpolate(Constant(4), V)

    f_vec = DolfinPrimalVector(f)
    g_vec = DolfinPrimalVector(g)

    return DolfinPrimalVectorSet([f_vec, g_vec])

def test_IfFunctionIsWrappedToPrimalVector_ThenTheNormIsCorrect(primal):
    assert abs(primal.norm() - sqrt(2**2 + 4**2)) < 1e-10

def test_IfFunctionIsWrappedToDualVector_ThenTheNormIsCorrect(primal):
    dual = primal.dual()
    assert abs(dual.primal_norm() - sqrt(2**2 + 4**2)) < 1e-10

def test_IfAVectorIsMappedToDualAndBack_ThenItHasNotChanged(primal):
    dual = primal.dual()
    primal2 = dual.primal()
    diff = (primal - primal2).norm()
    assert abs(diff) < 1e-10
