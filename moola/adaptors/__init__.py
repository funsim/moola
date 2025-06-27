from .numpy_vector import NumpyPrimalVector
from .numpy_vector import NumpyDualVector

from .dolfinx_vector import DolfinxPrimalVector
from .dolfinx_vector import DolfinxDualVector
from .dolfinx_vector_set import DolfinxPrimalVectorSet
from .dolfinx_vector_set import DolfinxDualVectorSet

from .dolfin_vector import DolfinPrimalVector
from .dolfin_vector import DolfinDualVector
from .dolfin_vector_set import DolfinPrimalVectorSet
from .dolfin_vector_set import DolfinDualVectorSet

# FIXME: Make these functions backend agnostic.
try:
    import dolfinx
    from .dolfinx_vector import RieszMap
    from .dolfinx_vector_set import RieszMapSet
except ModuleNotFoundError:
    from .dolfin_vector import RieszMap
    from .dolfin_vector_set import RieszMapSet



from .adaptor import convert_to_moola_dual_vector