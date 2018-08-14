from .dolfin_vector import DolfinPrimalVector, DolfinDualVector
from .dolfin_vector_set import DolfinDualVectorSet
from .numpy_vector import NumpyPrimalVector, NumpyDualVector


def convert_to_moola_dual_vector(x, y):
    """Convert `x` to a moola dual vector with primal vector `y`.

    """
    if isinstance(y, DolfinPrimalVector):
        r = DolfinDualVector(x, riesz_map=y.riesz_map)
    elif isinstance(y, NumpyPrimalVector):
        r = NumpyDualVector(x)
    else:
        r = DolfinDualVectorSet(
            [DolfinDualVector(xi, riesz_map=yi.riesz_map) for (xi, yi) in zip(x, y.vector_list)],
            riesz_map=y.riesz_map)

    return r

