"""
The optimization module is automatically imported by dolfin-adjoint 
"""

__version__ = '1.1.0+'
__author__  = 'Patrick Farrell and Simon Funke'
__credits__ = ['Patrick Farrell', 'Simon Funke', 'David Ham', 'Marie Rognes']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 's.funke09@imperial.ac.uk'

from armijo import ArmijoLineSearch
from strong_wolfe import StrongWolfeLineSearch
from fixed import FixedLineSearch
