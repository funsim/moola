from vector import Vector
import numpy as np

class NumpyVector(Vector):
    ''' An abstract implementation for vectors. '''

    def __init__(self, values):
        ''' Creates a new NumpyVector. The parameter values must be 
        a numpy array. '''
        self.vec = np.array(values)

    def __len__(self):
        ''' Returns the (local) length. '''
        return len(self.vec)

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.vec[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self[index] = value

    def scale(self, s):
        self.vec *= s

    def inner(self, vec):
        ''' Computes the inner product of the function and vec. ''' 
        np.dot(self.vec, vec)

    def norm(self, type="l2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        if type=="L1":
            return sum(abs(self.vec))
        elif type=="L2":
            return np.sqrt(sum(self.vec**2))

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        self.vec += a*x
