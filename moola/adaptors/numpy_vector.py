from moola.linalg import Vector
import numpy as np

class NumpyVector(Vector):
    ''' An implementation for vectors based on numpy arrays. '''

    def __init__(self, data):
        self.data = np.array(data)

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[index] = value

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data

    def scale(self, s):
        self.data *= s

    def dot(self, data):
        ''' Computes the dot product of the function and data. ''' 
        return float(np.dot(self.data, data))

    def inner(self, data):
        ''' Computes the inner product of the function and data. ''' 
        return float(np.dot(self.data, data))

    def norm(self, type="L2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        if type=="L1":
            return sum(abs(self.data))
        elif type=="L2":
            return np.sqrt(sum(self.data**2))
        else:
            raise ValueError, "Unkown norm"

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        self.data += a*x.data

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return len(self.data)

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data)

    def copy(self):
        return NumpyVector(self.data.copy())

class NumpyLinearFunctional(NumpyVector):

    def __call__(self, d):
        return self.dot(d)

    def riesz_representation(self):
        return self
