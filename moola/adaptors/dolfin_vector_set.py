from .dolfin_vector import DolfinVector, DolfinPrimalVector, DolfinDualVector
from moola.linalg import Vector
from moola.misc import events
from math import sqrt
from numpy import array, zeros, ndarray, eye

class RieszMapSet(object):

    def __init__(self, inner_product="l2", map_operator=None, inverse = "default"):
        
        if inner_product == "l2":
            self.riesz_map = 1
            self.riesz_inv = 1
            
        elif inner_product == "custom":
            from numpy import ndarray, equal
            from scipy.sparse.linalg import LinearOperator, aslinearoperator
            if not isinstance(map_operator, (ndarray, LinearOperator)) or not equal(*map_operator.shape):
                raise TypeError("only square numpy arrays are currently supported")
            self.riesz_map = aslinearoperator(map_operator)
            
            if inverse in ("default", "lu") and isinstance(map_operator, ndarray):
                from numpy.linalg import inv
                self.riesz_inv = aslinearoperator(inv(map_operator))
            
            else:
                self.riesz_inv = inverse
        
        self.inner_product = inner_product
                
    def primal_map(self, x, b):
        if isinstance(self.riesz_inv, ndarray):
            b[:] = self.riesz_inv.dot(x.vector_list)
        else:
            b[:] = self.riesz_inv * x.vector_list

    def dual_map(self, x):
        if isinstance(self.riesz_map, ndarray):
            b[:] = self.riesz_map.dot(x.vector_list)
        else:
            b[:] = self.riesz_map * x.vector_list

class IdentityMapSet(RieszMapSet):
    def __init__(self):
        RieszMapSet.__init__(self, inner_product = "l2")

class DolfinVectorSet(Vector):

    def __init__(self, vector_list, riesz_map = None):
        '''An implementation for set of vectors based on FEniCS data types.
        Currently supports only simple Riesz maps of numpy.ndarry type.

        Args:
            vector_list (list): A list with moola.DolfinVector
            
            riesz_map (array): An operator to be applied to the vector
            of controls prior to the individual control Riesz maps.

        '''

        for vec in vector_list:
            if not isinstance(vec, DolfinVector):
                raise ValueError("vector_list must be a list of DolfinVectors")
        self.vector_list = zeros(len(vector_list), dtype = object)
        self.vector_list[:] = vector_list
        
        if riesz_map == None:
            riesz_map = RieszMapSet("l2")
            
        self.riesz_map = riesz_map

    def __len__(self):
        return len(self.vector_list)

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
        if hasattr(self, "riesz_map"):
            return self.__class__(vector_list_cpy, riesz_map = self.riesz_map)
        else:
            return self.__class__(vector_list_cpy)


class DolfinPrimalVectorSet(DolfinVectorSet):
    """ A class for representing primal vectors. """
    def __init__(self, vector_list, riesz_map = None):
        for i, v in enumerate(vector_list):
            if not isinstance(v, DolfinPrimalVector):
                raise TypeError("Vector with index {} is not a DolfinPrimalVector.".format(i))
        DolfinVectorSet.__init__(self, vector_list, riesz_map = riesz_map)

    def dual(self):
        """ Returns the dual representation. """
        if self.riesz_map.inner_product == "l2":
            return DolfinDualVectorSet([vec.dual() for vec in self.vector_list],
                                       riesz_map = self.riesz_map)
        else:
            return DolfinDualVectorSet([vec.dual() for vec in self.riesz_map.riesz_map * self.vector_list],
                                       riesz_map = self.riesz_map)
            

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
    def __init__(self, vector_list, riesz_map = None):
        for i, v in enumerate(vector_list):
            if not isinstance(v, DolfinDualVector):
                raise TypeError("Vector with index {} is not a DolfinDualVector.".format(i))
        DolfinVectorSet.__init__(self, vector_list, riesz_map)

    def apply(self, primal):
        """ Applies the dual vector to a primal vector. """
        assert isinstance(primal, DolfinPrimalVectorSet)
        return sum([ss.apply(vec) for ss, vec in zip(self.vector_list, primal.vector_list)])

    def primal(self):
        """ Returns the primal representation. """
        #events.increment("Dual -> primal map")
        if self.riesz_map.inner_product == "l2":
            return DolfinPrimalVectorSet([vec.primal() for vec in self.vector_list], 
                                         riesz_map = self.riesz_map)
        else:
            primal_vecs = zeros(len(self), dtype = "object")
            primal_vecs[:] = [v.primal() for v in self.vector_list]
            return DolfinPrimalVectorSet(self.riesz_map.riesz_inv * primal_vecs,
                                         riesz_map = self.riesz_map)
            

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinLinearFunctional = DolfinDualVector
