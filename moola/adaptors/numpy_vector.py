from moola.linalg import Vector
from moola.misc import events
from math import sqrt
import numpy as np

class NumpyVector(Vector):
    ''' An implementation for vectors based on numpy arrays. '''

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[index] = value

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data

    def scale(self, s):
        self.data *= s

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        self.data += a*x.data

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return len(self.data)

    def zero(self):
        ''' Zeros the vector. '''
        self.data[:] = 0
        return self

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data)

    def copy(self):
        return self.__class__(self.data.copy())


class NumpyPrimalVector(NumpyVector):
    """ A class for representing primal vectors. """

    def dual(self):
        """ Returns the dual representation. """

        events.increment("Primal -> dual map")
        return NumpyDualVector(self.data.copy())

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, NumpyPrimalVector)
        events.increment("Inner product")

        return float(np.dot(self.data, vec.data))

    def norm(self):
        """ Computes the vector norm induced by the inner product. """

        return sqrt(self.inner(self))
    primal_norm = norm


class NumpyDualVector(NumpyVector):
    """ A class for representing dual vectors. """

    def apply(self, primal):
        """ Applies the dual vector to a primal vector. """
        assert isinstance(primal, NumpyPrimalVector)
        return float(np.dot(self.data, primal.data))
    
    def primal(self):
        """ Returns the primal representation. """
        events.increment("Dual -> primal map")

        return NumpyPrimalVector(self.data.copy())

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))


NumpyLinearFunctional = NumpyDualVector
