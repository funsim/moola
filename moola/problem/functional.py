class DerivativeOperator(object):

    def __init__(self, grad):
        self.grad = grad 

    def __call__(self, d):
        return self.grad.inner(d)

    def riesz_representation(self):
        return self.grad

    def __sub__(self, b):
        return ComposedDerivativeOperator([self, b], [1., -1.])

    def __lmul__(self, a):
        return ComposedDerivativeOperator([self], [a])
    __rmul__ = __lmul__


class ComposedDerivativeOperator(DerivativeOperator):
    '''  Represents linear composition of operators
            \sum_i s_i*op_i
    '''

    def __init__(self, ops, scales):
        self.ops = ops
        self.scales = scales

    def __call__(self, d):
        d = ([s * op(d) for (op, s) in zip(self.ops, self.scales)])
        return reduce(lambda x, y: x + y, d)

    def riesz_representation(self):
        d = [s * op.riesz_representation() for (op, s) in zip(self.ops, self.scales)]
        return reduce(lambda x, y: x + y, d)


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
        return DerivativeOperator(grad)


    def Hessian(self, x):
        ''' Returns the Hessian of the object functional at x as a function. '''
        raise NotImplementedError, "Functional.Hessian is not implemented." 
