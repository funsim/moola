class Vector(object):
    ''' An abstract implementation for vectors. '''

    def scale(self, s):
        raise NotImplementedError, "Vector.scale not implemented"

    def inner(self, f):
        ''' Computes the inner product of the function and f. ''' 
        raise NotImplementedError, "Vector.inner not implemented"

    def norm(self, type="l2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        raise NotImplementedError, "Vector.norm not implemented"

    def deep_copy(self):
        ''' Returns a deep copy. ''' 
        raise NotImplementedError, "Vector.norm not implemented"
    
    def axpy(self, a, f):
        ''' Adds a*f to the function. '''
        raise NotImplementedError, "Vector.axpy not implemented"

    def assign(self, x):
        ''' Assigns x to the function. '''
        raise NotImplementedError, "Vector.assign not implemented"
