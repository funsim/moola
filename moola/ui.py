from misc.infinity import inf

from problem import Problem
from problem import Functional

from algorithms import BFGS
from algorithms import NewtonCG
from algorithms import NonLinearCG
from algorithms import FletcherReeves
from algorithms import SteepestDescent

from adaptors.numpy_vector import NumpyPrimalVector
from adaptors.numpy_vector import NumpyDualVector
from adaptors.dolfin_vector import DolfinPrimalVector
from adaptors.dolfin_vector import DolfinDualVector
