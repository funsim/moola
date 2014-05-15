The Moola optimisation package
==============================

Moola implements a set of optimisation algorithms with a special focus on PDE-constrained optimisation problems.

You need to have FEniCS_ and dolfin-adjoint_ installed to run the tests.


.. _FEniCS: http://www.fenicsproject.org
.. _dolfin-adjoint: http://dolfin-adjoint.org


Update pypi Moola package
-------------------------

- Increase the version number on setup.py and moola/__init__.py
- Run `python setup.py register bdist_egg upload`  
