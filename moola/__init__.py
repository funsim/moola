"""
The Moola optimization module
"""

__version__ = '0.1.5'
__author__  = 'Simon Funke'
__credits__ = ['Simon Funke']
__license__ = 'LGPL-3'
__maintainer__ = 'Simon Funke'
__email__ = 's.funke09@imperial.ac.uk'

from . import linalg
from . import problem
from . import linesearch
from . import algorithms
from . import misc

from .ui import *
