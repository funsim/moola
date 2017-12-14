class Vector(object):
    ''' An abstract implementation for vectors. '''

    ############# To be implement by overloaded class #####################
    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        raise NotImplementedError("Vector.__getitem__ is not implemented")

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        raise NotImplementedError("Vector.__setitem__ is not implemented")

    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        raise NotImplementedError("Vector.array is not implemented")

    def set(self, array, local=True):
        ''' Sets the values of the vector to the values in the numpy.array. 
        If local=False, the global array must be returned in a distributed environment. '''
        raise NotImplementedError("Vector.set is not implemented")

    def scale(self, s):
        ''' Scales the vector by s. '''
        raise NotImplementedError("Vector.scale is not implemented")

    def axpy(self, a, x):
        ''' Adds a*x to the vector. '''
        raise NotImplementedError("Vector.axpy is not implemented")

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        raise NotImplementedError("Vector.__local_size__ is not implemented")

    def size(self):
        ''' Returns the (global) size of the vector. '''
        raise NotImplementedError("Vector.__size__ is not implemented")

    def has_petsc_support(self):
        ''' Returns True if the vector can be converted to a PETSc vector. '''
        raise NotImplementedError("Vector.has_petsc_support is not implemented")

    def to_petsc(self):
        ''' Returns the PETSc vector. Must only be implemented if has_petsc_support returns True. ''' 
        raise NotImplementedError("Vector.to_petsc is not implemented")


    ################# Default implementations #####################
    def __init__(self, data):
        ''' Creates a new Vector.'''
        self.data = data

    def __lmul__(self, a):
        ''' Scales the vector by a. '''
        c = self.copy()
        c.scale(a)
        return c

    __rmul__ = __lmul__

    def __add__(self, v):
        ''' Adds v to the vector. '''
        c = self.copy()
        c.axpy(1.0, v)
        return c

    def __sub__(self, v):
        ''' Adds v to the vector. '''
        c = self.copy()
        c.axpy(-1.0, v)
        return c

    def __neg__(self):
        ''' Negates the vector. '''
        c = self.copy()
        c.scale(-1)
        return c

    def __len__(self):
        ''' Returns the (local) size. '''
        return self.local_size()

    def copy(self):
        ''' Returns a deep-copy of the vector. '''
        d = self.data.__class__(self.data)
        return Vector(d)

    def assign(self, x):
        ''' Copies the values from x to the function. '''
        self.zero()
        self.axpy(1.0, x)
    
