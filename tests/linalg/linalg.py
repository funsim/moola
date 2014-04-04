from moola import *
from dolfin import *

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)

g = interpolate(Constant(2), V)

pg = DolfinPrimalVector(g)
assert abs(pg.norm() - 2) < 1e-10

dg = pg.dual()
assert abs(dg.primal_norm() - 2) < 1e-10

pdg = dg.primal()
diff = (pg - pdg).norm()
assert abs(diff) < 1e-10
