from moola.linalg import Vector
import dolfin 
from math import sqrt

class DolfinVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data):
        ''' Creates a new DolfinVector with a deep-copy of the 
        underlying data. The parameter 'data' must be 
        a DolfinVector or a numpy.array. '''
        self.data = data 

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

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        v = self.data.vector()
        v.axpy(a, x.data.vector())

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return self.data.vector().local_size()

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return self.data.vector().size()

    def copy(self):
        return self.__class__(self.data.copy(deepcopy=True))


class DolfinPrimalVector(DolfinVector):
    """ A class for representing primal vectors. """

    def dual(self):
        """ Returns the dual representation. """

        if isinstance(self.data, dolfin.Function):

            V = self.data.function_space()
            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)
            M = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)

            primal_vec = M * self.data.vector()
            primal = dolfin.Function(V, primal_vec)

            return DolfinDualVector(primal)
        else:
            return self

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, DolfinPrimalVector)

        return dolfin.assemble(dolfin.inner(self.data, vec.data)*dolfin.dx)


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
        if isinstance(self.data, dolfin.Function):

            V = self.data.function_space()
            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)
            M = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)

            dual = dolfin.Function(V)
            dolfin.solve(M, dual.vector(), self.data.vector())

            return DolfinPrimalVector(dual)
        else:
            return self

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinLinearFunctional = DolfinDualVector
