from moola.linalg import Vector
from moola.misc import events
from math import sqrt

"""A (limited) copy of the Dolfin vector adapter."""

class IdentityMap(object):
    def primal_map(self, x, b):
        x.assign(0.0) # zero out primal result vector
        x.axpy(1, b) # assign dual vector verbatim

class RieszMap(object):
    def __init__(self, V, inner_product="L2"):
        import firedrake

        self.V = V

        u = firedrake.TrialFunction(V)
        v = firedrake.TestFunction(V)

        default_forms = {"L2": firedrake.inner(u, v) * firedrake.dx}

        if inner_product not in default_forms:
            raise Exception("unsupported RieszMap inner product")

        self.form = default_forms[inner_product]
        self.map_operator = firedrake.assemble(self.form)

    def primal_map(self, x, b):
        import firedrake
        firedrake.solve(self.map_operator, x, b)

    def dual_map(self, x):
        # because we can't (that I know of) work with the assembled matrix,
        # we just have to work with the form directly
        def _f(v):
            import firedrake
            return firedrake.assemble(self.form(x, v, coefficients={}))

        return _f

class FiredrakeVector(Vector):
    def __init__(self, data, inner_product="L2", riesz_map=None):

        if riesz_map is None:
            if inner_product == "l2":
                riesz_map = IdentityMap()
            else:
                fn_space = data.function_space()
                riesz_map = RieszMap(fn_space, inner_product)

        self.riesz_map = riesz_map
        self.data = data

    def scale(self, s):
        self.data *= s

    def copy(self):
        return self.__class__(self.data.copy(deepcopy=True), riesz_map=self.riesz_map)

    def zero(self):
        self.data.assign(0.0)
        return self

    def axpy(self, a, x):
        self.data += a * x.data

class FiredrakePrimalVector(FiredrakeVector):
    def inner(self, vec):
        v = self.riesz_map.dual_map(self.data)
        return v(vec.data)

    def norm(self):
        return sqrt(self.inner(self))

class FiredrakeDualVector(FiredrakePrimalVector):
    def apply(self, primal):
        assert isinstance(primal, FiredrakePrimalVector)
        return self.data.vector().inner(primal.data.vector())

    def primal(self):
        events.increment("Dual -> primal map")

        import firedrake

        if isinstance(self.data, firedrake.Function):
            V = self.data.function_space()
            dual = firedrake.Function(V)
            self.riesz_map.primal_map(dual.vector(), self.data.vector())
            return FiredrakePrimalVector(dual, riesz_map=self.riesz_map)


        raise NotImplementedError("non-function dual->primal map")

    def primal_norm(self):
        return sqrt(self.apply(self.primal()))

