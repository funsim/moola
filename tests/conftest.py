import sys
import dolfin
import dolfin_adjoint

# Automatically parallelize over all cpus
#def pytest_cmdline_preparse(args):
#    if 'xdist' in sys.modules: # pytest-xdist plugin
#        import multiprocessing
#        num = multiprocessing.cpu_count()
#        args[:] = ["-n", str(num)] + args


default_params = dolfin.parameters.copy()
def pytest_runtest_setup(item):
    """ Hook function which is called before every test """

    # Reset dolfin parameter dictionary
    dolfin.parameters.update(default_params)

    # Reset adjoint state
    dolfin_adjoint.adj_reset()
