from moola.linalg import Vector
import dolfin 

class DolfinVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data):
        ''' Creates a new DolfinVector with a deep-copy of the 
        underlying data. The parameter 'data' must be 
        a DolfinVector or a numpy.array. '''
        self.data = data 

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
        v = self.data.vector()
        v *= s

    def inner(self, v):
        ''' Computes the inner product of the function and data. ''' 
        r = dolfin.inner(self.data, v.data)*dolfin.dx
        return dolfin.assemble(r)

    def norm(self, type="L2"):
        ''' Computes the function norm. Valid types are "L1", "L2", and "Linf"''' 
        if type=="L2":
            return dolfin.norm(self.data, "L2")
        elif type=="Linf":
            return dolfin.norm(self.data, "linf")
        else:
            raise NotImplementedError, "Unkown norm"

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        v = self.data.vector()
        v.axpy(a, x.data.vector())

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return self.data.vector().local_size()

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return self.data.vector().size()

    def copy(self):
        return self.__class__(self.data.copy(deepcopy=True))

class DolfinLinearFunctional(DolfinVector):

    def __call__(self, d):
        return self.data.vector().inner(d.data.vector())
    
    apply = __call__

    def riesz_representation(self):
        if isinstance(self.data, dolfin.Function):

            V = self.data.function_space()
            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)
            M = dolfin.assemble(dolfin.inner(u, v)*dolfin.dx)

            proj = dolfin.Function(V)
            dolfin.solve(M, proj.vector(), self.data.vector())

            return DolfinVector(proj)
        else:
            return self

