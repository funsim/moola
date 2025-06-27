from moola.linalg import Vector
from moola.misc import events
from math import sqrt


class IdentityMap():

    def primal_map(self, x, b):
        x.zero()
        x.axpy(1, b)

    def dual_map(self, x):
        return x.copy()

class RieszMap():

    def __init__(self, V, inner_product="L2", map_operator=None, inverse = "default",
                 jit_options=None, form_compiler_options=None):
        self.V = V
        from petsc4py import PETSc
        import ufl
        import dolfinx.fem.petsc
        if inner_product!="custom":
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)


            default_forms = {"L2":   ufl.inner(u, v)*ufl.dx,
                                "H0_1": ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx,
                                "H1":  (ufl.inner(u, v) + ufl.inner(ufl.grad(u),
                                                                        ufl.grad(v)))*ufl.dx,
            }

            form = dolfinx.fem.form(default_forms[inner_product], jit_options=jit_options,form_compiler_options=form_compiler_options)
            map_operator = dolfinx.fem.petsc.assemble_matrix(form)
            map_operator.assemble()
        self.map_operator = map_operator
        self._P = None
        if inverse in ("default", "lu"):
            self.petsc_options = {"ksp_type": "preonly", "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "ksp_error_if_not_converged": True}
        elif inverse == "jacobi":
            self.petsc_options = {"ksp_type": "preonly", "pc_type": "jacobi",
                            "ksp_error_if_not_converged": True}
        elif inverse == "sor":
            self.petsc_options = {"ksp_type": "preonly", "pc_type": "sor",
                            "ksp_error_if_not_converged": True}

        elif inverse == "amg":
            self.petsc_options = {"ksp_type": "preonly", "pc_type": "hypre"}

        elif isinstance(inverse, PETSc.Mat):
            self.petsc_options = {"skp_type": "preonly", "pc_type": "mat"}
            self._P = inverse
        else:
            raise RuntimeError(f"Unknown inverse type {inverse}. ")
        self.solver_type = inverse
    def primal_map(self, x, b):
        from dolfinx_adjoint.petsc_utils import solve_linear_problem
        solve_linear_problem(self.map_operator, x, b, petsc_options=self.petsc_options, P=self._P)


    def dual_map(self, x):
        return self.map_operator * x


class DolfinxVector(Vector):
    ''' An implementation for vectors based on Dolfin data types. '''

    def __init__(self, data, inner_product="L2", riesz_map=None):
        ''' Wraps the Dolfinx object `data` (Function) in a moola.DolfinxVector object. '''

        if riesz_map is None:
            if inner_product=="l2":
                riesz_map = IdentityMap()
            else:
                fn_space = data.function_space
                riesz_map = RieszMap(fn_space, inner_product)
        self.riesz_map = riesz_map
        self.data = data
        self.version = 0

    def bump_version(self):
        self.version += 1

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data.x.array[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data.x.array[index] = value

    def array(self):
        ''' Returns the vector as a numpy.array object. If local=False, the
        global array must be returned in a distributed environment. '''
        return self.data.x.array

    def scale(self, s):
        v = self.data.x.array
        v *= s
        self.bump_version()

    def __hash__(self):
        ''' Returns a hash of the vector '''
        return hash(self.data)*100 + self.version

    def axpy(self, a, x):
        ''' Adds a*x to the function. '''
        assert self.__class__ == x.__class__
        v = self.data.x.array
        v[:] += a * x.data.x.array[:]
        self.bump_version()

    def zero(self):
        ''' Zeros the function. '''
        self.data.x.array[:] = 0
        self.bump_version()
        return self

    def local_size(self):
        ''' Returns the (local) size of the vector. '''
        return self.data.index_map.size_local* self.data.x.block_size

    def size(self):
        ''' Returns the (gobal) size of the vector. '''
        return self.data.index_map.size_global* self.data.x.block_size

    def copy(self):
        # FIXME: As we don't have a copy method for the dolfinx Function, this is a bit tedious.
        from dolfinx_adjoint import Function
        from dolfinx_adjoint.blocks.assembly import _vector
        func = Function(self.data.function_space, annotate=True)
        func.x.array[:] = self.data.x.array.copy()
        tt =  self.__class__(func,
                              riesz_map=self.riesz_map)
        return tt


class DolfinxPrimalVector(DolfinxVector):
    """ A class for representing primal vectors. """


    def dual(self):
        """ Returns the dual representation. """

        events.increment("Primal -> dual map")
        import dolfinx.fem
        if isinstance(self.data, dolfinx.fem.Function):
            V = self.data.function_space

            dual_vec = self.riesz_map.dual_map(self.data.x)
            dual = dolfinx.fem.Function(V, dual_vec)

            return DolfinxDualVector(dual, riesz_map=self.riesz_map)
        else:
            return self

    def inner(self, vec):
        """ Computes the inner product with vec. """
        assert isinstance(vec, DolfinxPrimalVector)
        events.increment("Inner product")

        v = self.riesz_map.dual_map(self.data.x)
        import dolfinx.cpp
        return dolfinx.cpp.la.inner_product(v._cpp_object, self.data.x._cpp_object)

    def norm(self):
        """ Computes the vector norm induced by the inner product. """

        return sqrt(self.inner(self))
    primal_norm = norm


class DolfinxDualVector(DolfinxVector):
    """ A class for representing dual vectors. """

    def apply(self, primal):
        """ Applies the dual vector to a primal vector. """
        assert isinstance(primal, DolfinxPrimalVector)
        import dolfinx.cpp
        return dolfinx.cpp.la.inner_product(self.data.x._cpp_object, primal.data.x._cpp_object)
    


    def primal(self):
        """ Returns the primal representation. """
        events.increment("Dual -> primal map")
        import dolfinx
        if isinstance(self.data, dolfinx.fem.Function):
            V = self.data.function_space

            dual = dolfinx.fem.Function(V)
            self.riesz_map.primal_map(dual.x, self.data.x)

            return DolfinxPrimalVector(dual, riesz_map=self.riesz_map)
        else:
            return self

    def primal_norm(self):
        """ Computes the norm of the primal representation. """

        return sqrt(self.apply(self.primal()))

DolfinxLinearFunctional = DolfinxDualVector
