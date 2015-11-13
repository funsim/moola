from dolfin_vector import DolfinVector, DolfinPrimalVector, DolfinDualVector
from moola.linalg import Vector
from moola.misc import events
from math import sqrt

class DolfinVectorSet(Vector):

    def __init__(self, vector_list):
        ''' An implementation for set of vectors based on FEniCS data types.

        Args:
            vector_list (list): A list with moola.DolfinVector
        '''

        for vec in vector_list:
            if not isinstance(vec, DolfinVector):
                from IPython import embed; embed()
                raise ValueError, "vector_list must be a list of DolfinVectors"
        self.vector_list = vector_list

    def __getitem__(self, index):
        ''' Returns the subvector with given index. '''
        return self.vector_list[index]

    def __setitem__(self, index, value):
        ''' Sets the subvector with the given index. '''
        self.vector_list[index] = value

    @property
    def data(self):
        return [vec.data for vec in self.vector_list]

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the
        global array must be returned in a distributed environment. '''
        #return self.data.vector().array()
        raise NotImplementedError

    def scale(self, s):
        [vec.scale(s) for vec in self.vector_list]

    def __hash__(self):
        ''' Returns a hash of the vector '''
        hashes = tuple([hash(vec) for vec in self.vector_list])
        return hash(hashes)

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        assert self.__class__ == x.__class__
        return [vec.axpy(a, xx) for vec, xx in zip(self.vector_list, x.vector_list)]


    def zero(self):
        ''' Zeros the function. '''
        [vec.zero() for vec in self.vector_list]
        return self

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        #return self.data.vector().local_size()
        raise NotImplementedError

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return sum([vec.size() for vec in self.vector_list])

    def copy(self):
        vector_list_cpy = [vec.copy() for vec in self.vector_list]
        return self.__class__(vector_list_cpy)


class DolfinPrimalVectorSet(DolfinVectorSet):
    """ A class for representing primal vectors. """


    def dual(self):
        """ Returns the dual representation. """
        return DolfinDualVectorSet([vec.dual() for vec in self.vector_list])

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, DolfinPrimalVectorSet)
        events.increment("Inner product")

        return sum([ss.inner(vec) for ss, vec in zip(self.vector_list, vec.vector_list)])

    def norm(self):
        """ Computes the vector norm induced by the inner product. """

        return sqrt(self.inner(self))
    primal_norm = norm


class DolfinDualVectorSet(DolfinVectorSet):
    """ A class for representing dual vectors. """

    def apply(self, primal):
        """ Applies the dual vector to a primal vector. """
        assert isinstance(primal, DolfinPrimalVectorSet)
        return sum([ss.apply(vec) for ss, vec in zip(self.vector_list, primal.vector_list)])

    def primal(self):
        """ Returns the primal representation. """
        events.increment("Dual -> primal map")

        return DolfinPrimalVectorSet([vec.primal() for vec in self.vector_list])

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinLinearFunctional = DolfinDualVector
