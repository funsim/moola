from moola.linalg import Vector
from moola.misc import events
import dolfin 
from math import sqrt

class DolfinVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data):
        ''' Creates a new DolfinVector with a deep-copy of the 
        underlying data. The parameter 'data' must be 
        a DolfinVector or a numpy.array. '''
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
        return self.__class__(self.data.copy(deepcopy=True))


class Cache(object):
    M = None
    M_solver = None

    def mass_matrix(self, V):
        if self.M is None or True:
            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)
            M = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)
            self.M = M
        return self.M

    def mass_solve(self, V, x, b):
        if self.M_solver is None or True:
            M = self.mass_matrix(V)
            M_solver = dolfin.LUSolver(M)
            M_solver.parameters["reuse_factorization"] = True
            self.M_solver = M_solver

        self.M_solver.solve(x, b)


cache = Cache()

class DolfinPrimalVector(DolfinVector):
    """ A class for representing primal vectors. """

    def dual(self):
        """ Returns the dual representation. """

        events.increment("Primal -> dual map")
        if isinstance(self.data, dolfin.Function):
            V = self.data.function_space()

            primal_vec = cache.mass_matrix(V) * self.data.vector()
            primal = dolfin.Function(V, primal_vec)

            return DolfinDualVector(primal)
        else:
            return self

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, DolfinPrimalVector)
        events.increment("Inner product")

        V = self.data.function_space()
        v = cache.mass_matrix(V) * self.data.vector()
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

        if isinstance(self.data, dolfin.Function):
            V = self.data.function_space()

            dual = dolfin.Function(V)
            cache.mass_solve(V, dual.vector(), self.data.vector())

            return DolfinPrimalVector(dual)
        else:
            return self

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinLinearFunctional = DolfinDualVector
