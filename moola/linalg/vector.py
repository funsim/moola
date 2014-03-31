class Vector(object):
    ''' An abstract implementation for vectors. '''

    ############# To be implement by overloaded class #####################
    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        raise NotImplementedError, "Vector.__getitem__ is not implemented"

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        raise NotImplementedError, "Vector.__setitem__ is not implemented"

    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        raise NotImplementedError, "Vector.array is not implemented"

    def set(self, array, local=True):
        ''' Sets the values of the vector to the values in the numpy.array. 
        If local=False, the global array must be returned in a distributed environment. '''
        raise NotImplementedError, "Vector.set is not implemented"

    def scale(self, s):
        ''' Scales the vector by s. '''
        raise NotImplementedError, "Vector.scale is not implemented"

    def inner(self, f):
        ''' Computes the inner product of f and itself. ''' 
        raise NotImplementedError, "Vector.inner is not implemented"

    def norm(self, type="l2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        raise NotImplementedError, "Vector.norm is not implemented"

    def axpy(self, a, x):
        ''' Adds a*x to the vector. '''
        raise NotImplementedError, "Vector.axpy is not implemented"
    
    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        raise NotImplementedError, "Vector.__local_size__ is not implemented"

    def size(self):
        ''' Returns the (global) size of the vector. '''
        raise NotImplementedError, "Vector.__size__ is not implemented"

    def has_petsc_support(self):
        ''' Returns True if the vector can be converted to a PETSc vector. '''
        raise NotImplementedError, "Vector.has_petsc_support is not implemented"

    def to_petsc(self):
        ''' Returns the PETSc vector. Must only be implemented if has_petsc_support returns True. ''' 
        raise NotImplementedError, "Vector.to_petsc is not implemented"


    ############# Standard methods  #####################
    def __init__(self):
        ''' Creates a new Vector with (a deep-copy of) the provided data. '''
        self.data = data

    def __mul__(self, a):
        ''' Scales the vector by a. '''
        self.scale(a)

    def __add__(self, v):
        ''' Adds v to the vector. '''
        self.axpy(1.0, v)

    def __len__(self):
        ''' Returns the (local) size. '''
        return self.local_size()

