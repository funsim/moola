class Functional(object):

    def __call__(self, x):
        ''' Evaluates the object functional at x. '''
        raise NotImplementedError, "Functional.__call__ is not implemented." 

    def gradient(self, x):
        ''' Evaluates the gradient of the object functional at x. 
            The returned gradient is the Riez representation of the 
            derivative, i.e. is a vector in the same space.
         '''
        raise NotImplementedError, "Functional.gradient is not implemented." 

    def derivative(self, x):
        ''' Returns the derivative operator at x. '''
        grad = self.gradient(x)
        def deriv(d):
            return grad.inner(d)

        return deriv

    def Hessian(self, x):
        ''' Returns the Hessian of the object functional at x as a function. '''
        raise NotImplementedError, "Functional.Hessian is not implemented." 

class ObjectiveFunctional(Functional):
    pass
