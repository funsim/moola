"""
Line search methods for the Moola optimisation module.
"""

__author__  = 'Simon Funke'
__credits__ = ['Simon Funke']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 's.funke09@imperial.ac.uk'

from .armijo import ArmijoLineSearch
from .strong_wolfe import StrongWolfeLineSearch
from .fixed import FixedLineSearch
from .hager_zhang import HagerZhangLineSearch
