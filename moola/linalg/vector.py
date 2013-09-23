class Vector(object):
    ''' An abstract implementation for vectors. '''

    def __init__(self):
        ''' Creates a new Vector with (a deep-copy of) the provided data. '''
        self.data = data

    def __len__(self):
        ''' Returns the (local) length. '''
        raise NotImplementedError, "Vector.__len__ is not implemented"

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        raise NotImplementedError, "Vector.__getitem__ is not implemented"

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        raise NotImplementedError, "Vector.__setitem__ is not implemented"

    def scale(self, s):
        raise NotImplementedError, "Vector.scale is not implemented"

    def inner(self, f):
        ''' Computes the inner product of the function and f. ''' 
        raise NotImplementedError, "Vector.inner is not implemented"

    def norm(self, type="l2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        raise NotImplementedError, "Vector.norm is not implemented"

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        raise NotImplementedError, "Vector.axpy is not implemented"
