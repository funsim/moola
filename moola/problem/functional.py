class Functional(object):

    def __eval__(self, x):
        ''' Evaluates the object functional at x. '''
        raise NotImplementedError, "Functional.__eval__ is not implemented." 

    def gradient(self, x):
        ''' Evaluates the gradient of the object functional at x. 
            The returned gradient is the Riez representation of the 
            functional derivative at x, i.e. is a vector in the same space
            as x. '''
        raise NotImplementedError, "Functional.gradient is not implemented." 

class ObjectiveFunctional(Functional):
    pass
