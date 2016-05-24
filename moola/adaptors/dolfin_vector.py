from moola.linalg import Vector
from moola.misc import events
from math import sqrt
import sys
if 'backend' not in sys.modules:
    import dolfin
    sys.modules['backend'] = dolfin
backend = sys.modules['backend']


class IdentityMap(object):

    def primal_map(self, x, b):
        x.zero()
        x.axpy(1, b)

    def dual_map(self, x):
        return x.copy()


class RieszMap(object):

    def __init__(self, V, inner_product="L2", map_operator=None, inverse="default"):
        self.V = V

        if inner_product is not "custom":
            u = backend.TrialFunction(V)
            v = backend.TestFunction(V)

            default_forms = {"L2":   backend.inner(u, v)*backend.dx,
                             "H0_1": backend.inner(backend.grad(u), backend.grad(v))*backend.dx,
                             "H1":  (backend.inner(u, v) + backend.inner(backend.grad(u), backend.grad(v)))*backend.dx,
                             }

            form = default_forms[inner_product]
            map_operator = backend.assemble(form)
        self.map_operator = map_operator
        if backend.__name__ == "firedrake":
            solver_parameters = {"ksp_type": "preonly"}
            if isinstance(inverse, backend.Matrix):
                solver_parameters["pc_type"] = "mat"
                self.map_solver = backend.LinearSolver(
                    self.map_operator, inverse,
                    solver_parameters=solver_parameters)
            elif isinstance(inverse, str):
                if inverse == "default":
                    inverse == "lu"
                if inverse == "amg":
                    inverse == "hypre"

                solver_parameters["pc_type"] = inverse
                self.map_solver = backend.LinearSolver(
                    self.map_operator,
                    solver_parameters=solver_parameters)
            else:
                raise NotImplementedError(
                    "Don't know how to deal with inverse %s" % inverse)
        else:
            if inverse in ("default", "lu"):
                self.map_solver = backend.LUSolver(self.map_operator)
                self.map_solver.parameters["reuse_factorization"] = True

            elif inverse == "jacobi":
                self.map_solver = backend.PETScKrylovSolver()
                self.map_solver.set_operator(self.map_operator)
                self.map_solver.ksp().setType("preonly")
                self.map_solver.ksp().getPC().setType("jacobi")

            elif inverse == "sor":
                self.map_solver = backend.PETScKrylovSolver()
                self.map_solver.set_operator(self.map_operator)
                self.map_solver.ksp().setType("preonly")
                self.map_solver.ksp().getPC().setType("sor")

            elif inverse == "amg":
                self.map_solver = backend.PETScKrylovSolver()
                self.map_solver.set_operator(self.map_operator)
                self.map_solver.ksp().setType("preonly")
                self.map_solver.ksp().getPC().setType("hypre")
            elif isinstance(inverse, backend.GenericMatrix):
                self.map_solver = backend.PETScKrylovSolver()
                self.map_solver.set_operators(self.map_operator, inverse)
                self.map_solver.ksp().setType("preonly")
                self.map_solver.ksp().getPC().setType("mat")
            else:
                self.map_solver = inverse
                self.solver_type = inverse

    def primal_map(self, x, b):
        self.map_solver.solve(x, b)

    def dual_map(self, x):
        return self.map_operator * x


class DolfinVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data, inner_product="L2", riesz_map=None):
        ''' Wraps the Dolfin object `data` in a moola.DolfinVector object. '''

        if riesz_map is None:
            if inner_product == "l2":
                riesz_map = IdentityMap()
            else:
                fn_space = data.function_space()
                riesz_map = RieszMap(fn_space, inner_product)

        self.riesz_map = riesz_map
        self.data = data
        self.version = 0

    def bump_version(self):
        self.version += 1

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data.vector()[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data.vector()[index] = value

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the
        global array must be returned in a distributed environment. '''
        return self.data.vector().array()

    def scale(self, s):
        v = self.data.vector()
        v *= s
        self.bump_version()

    def __hash__(self):
        ''' Returns a hash of the vector '''
        return hash(self.data)*100 + self.version

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        assert self.__class__ == x.__class__

        v = self.data.vector()
        v.axpy(a, x.data.vector())
        self.bump_version()

    def zero(self):
        ''' Zeros the function. '''
        v = self.data.vector()
        v.zero()
        self.bump_version()
        return self

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return self.data.vector().local_size()

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return self.data.vector().size()

    def copy(self):
        return self.__class__(self.data.copy(deepcopy=True),
                              riesz_map=self.riesz_map)


class DolfinPrimalVector(DolfinVector):
    """ A class for representing primal vectors. """

    def dual(self):
        """ Returns the dual representation. """

        events.increment("Primal -> dual map")
        if isinstance(self.data, backend.Function):
            V = self.data.function_space()

            dual_vec = self.riesz_map.dual_map(self.data.vector())
            dual = backend.Function(V, dual_vec)

            return DolfinDualVector(dual, riesz_map=self.riesz_map)
        else:
            return self

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, DolfinPrimalVector)
        events.increment("Inner product")

        v = self.riesz_map.dual_map(self.data.vector())
        return v.inner(self.data.vector())

    def norm(self):
        """ Computes the vector norm induced by the inner product. """

        return sqrt(self.inner(self))
    primal_norm = norm


class DolfinDualVector(DolfinVector):
    """ A class for representing dual vectors. """

    def apply(self, primal):
        """ Applies the dual vector to a primal vector. """
        assert isinstance(primal, DolfinPrimalVector)
        return self.data.vector().inner(primal.data.vector())

    def primal(self):
        """ Returns the primal representation. """
        events.increment("Dual -> primal map")

        if isinstance(self.data, backend.Function):
            V = self.data.function_space()

            dual = backend.Function(V)
            self.riesz_map.primal_map(dual.vector(), self.data.vector())

            return DolfinPrimalVector(dual, riesz_map=self.riesz_map)
        else:
            return self

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinLinearFunctional = DolfinDualVector
