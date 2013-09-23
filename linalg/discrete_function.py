class DiscreteFunction(object):
    ''' An abstract implementation for functions of a finite dimensional space. '''

    def scale(self, s):
        raise NotImplementedError, "DiscreteFunction.scale not implemented"

    def inner(self, f):
        ''' Computes the inner product of the function and f. ''' 
        raise NotImplementedError, "DiscreteFunction.inner not implemented"

    def norm(self, type="l2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        raise NotImplementedError, "DiscreteFunction.norm not implemented"

    def deep_copy(self):
        ''' Returns a deep copy. ''' 
        raise NotImplementedError, "DiscreteFunction.norm not implemented"
    
    def axpy(self, a, f):
        ''' Adds a*f to the function. '''
        raise NotImplementedError, "DiscreteFunction.axpy not implemented"

    def assign(self, x):
        ''' Assigns x to the function. '''
        raise NotImplementedError, "DiscreteFunction.assign not implemented"
