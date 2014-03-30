from moola.vector import Vector
import dolfin 

class DolfinVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data):
        ''' Creates a new DolfinVector with a deep-copy of the 
        underlying data. The parameter 'data' must be 
        a DolfinVector or a numpy.array. '''
        if isinstance(data, DolfinVector):
            data = data.data

        self.data = data.__class__(data) 

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data.vector()[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data.vector()[index] = value

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data.vector().array()

    def scale(self, s):
        self.data[:] = s*self.data.array()

    def inner(self, data):
        ''' Computes the inner product of the function and data. ''' 
        return dolfin.inner(self.data, data)*dolfin.dx

    def norm(self, type="L2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        elif type=="L2":
            return dolfin.norm(self.data, "L2")
        elif type=="Linf":
            return dolfin.norm(self.data, "linf")
        else:
            raise NotImplementedError, "Unkown norm"

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        self.data.axpy(a, x)

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return self.data.vector().local_size()

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return self.data.vector().size()

