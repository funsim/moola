from problem import Problem, Functional
from linalg import Vector, LinearFunctional
from algorithms import SteepestDescent, FletcherReeves, BFGS, NonLinearCG
from misc.infinity import inf

from adaptors.numpy_vector import NumpyPrimalVector, NumpyDualVector
from adaptors.dolfin_vector import DolfinPrimalVector, DolfinDualVector

# Deprecated imports
from adaptors.numpy_vector import NumpyPrimalVector, NumpyDualVector, NumpyLinearFunctional
from adaptors.dolfin_vector import DolfinVector, DolfinLinearFunctional, DolfinPrimalVector, DolfinDualVector
