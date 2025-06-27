from dolfin import *
from moola import *
import pytest

@pytest.fixture
def f():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    return f

def test_IfVectorIsScale_ThenHashChanges(f):
    pf = DolfinPrimalVector(f)
    pf_hash0 = hash(pf)
    assert pf.version == 0
    pf.scale(-1)
    assert pf.version == 1
    pf_hash1 = hash(pf)
    assert pf_hash0 != pf_hash1

def test_IfVectorIsScaled_ThenDolfinHashDoesNotChange(f):
    pf = DolfinPrimalVector(f)
    f_hash0 = hash(f.vector())
    pf.scale(-1)
    f_hash1 = hash(f.vector())
    assert f_hash0 == f_hash1

def test_IfVectorIsPassedIntoFunction_ThenHashDoesNotChange(f):

    def function_hash(obj):
        return hash(obj)

    pf = DolfinPrimalVector(f)
    assert function_hash(f) == hash(f)
    assert function_hash(pf) == hash(pf)

def test_IfTwoFunctionsAreAllocated_ThenHashesAreDifferent(f):
    g = Function(f.function_space())
    assert hash(f) != hash(g)
