import dolfin
from dolfin import MPI 
from dolfin_adjoint import constant 
from ..reduced_functional_numpy import ReducedFunctionalNumPy, get_global, set_local
from ..reduced_functional import ReducedFunctional
import numpy as np
import sys

def serialise_bounds(rf_np, bounds):
    ''' Converts bounds to an array of (min, max) tuples and serialises it in a parallel environment. ''' 

    if len(np.array(bounds).shape) == 1:
        bounds = np.array([[b] for b in bounds])

    if len(bounds) != 2:
        raise ValueError, "The 'bounds' parameter must be of the form [lower_bound, upper_bound] for one parameter or [ [lower_bound1, lower_bound2, ...], [upper_bound1, upper_bound2, ...] ] for multiple parameters."

    bounds_arr = [[], []]
    for i in range(2):
        for j in range(len(bounds[i])):
            bound = bounds[i][j]
            if type(bound) in [int,  float, np.int32, np.int64, np.float32, np.float64]:
                bound_len = len(get_global(rf_np.parameter[j].data()))
                const_bound = bound*np.ones(bound_len)

                if rf_np.in_euclidian_space:
                   # Convert the bounds into Euclidian space if necessary
                   if rf_np.has_cholmod:
                       const_bound = rf_np.sqD*rf_np.factor.L_D()[0].transpose()*rf_np.factor.apply_P(const_bound)
                   else:
                       const_bound = np.dot(rf_np.LT, const_bound)


                bounds_arr[i] += const_bound.tolist()
            else:
                bounds_arr[i] += rf_np.obj_to_array(bound).tolist()

    # Transpose and return the array to get the form [ [lower_bound1, upper_bound1], [lower_bound2, upper_bound2], ... ] 
    return np.array(bounds_arr).T

def minimize_scipy_generic(rf_np, method, bounds = None, **kwargs):
    ''' Interface to the generic minimize method in scipy '''

    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError:
        print "**************** Deprecated warning *****************"
        print "You have an old version of scipy (<0.11). This version is not supported by dolfin-adjoint."
        raise ImportError

    if method in ["Newton-CG"]:
        forget = None
    else:
        forget = False

    m = [p.data() for p in rf_np.parameter]
    m_global = rf_np.obj_to_array(m)
    J = rf_np.__call__
    dJ = lambda m: rf_np.derivative(m, taylor_test=dolfin.parameters["optimization"]["test_gradient"], 
                                       seed=dolfin.parameters["optimization"]["test_gradient_seed"],
                                       forget=forget)
    H = rf_np.hessian

    if not "options" in kwargs:
        kwargs["options"] = {}
    if MPI.process_number() != 0:
        # Shut up all processors except the first one.
        kwargs["options"]["disp"] = False
    else:
        # Print out progress information by default
        if not "disp" in kwargs["options"]:
            kwargs["options"]["disp"] = True

    # Make the default SLSLQP options more verbose
    if method == "SLSQP" and "iprint" not in kwargs["options"]:
        kwargs["options"]["iprint"] = 2

    # For gradient-based methods add the derivative function to the argument list 
    if method not in ["COBYLA", "Nelder-Mead", "Anneal", "Powell"]:
        kwargs["jac"] = dJ

    # For Hessian-based methods add the Hessian action function to the argument list
    if method in ["Newton-CG"]:
        kwargs["hessp"] = H

    if method=="basinhopping":
        try:
            from scipy.optimize import basinhopping
        except ImportError:
            print "**************** Outdated scipy version warning *****************"
            print "The basin hopping optimisation algorithm requires scipy >= 0.12."
            raise ImportError

        del kwargs["options"]
        del kwargs["jac"]
        kwargs["minimizer_kwargs"]["jac"]=dJ

        if "bounds" in kwargs["minimizer_kwargs"]:
          kwargs["minimizer_kwargs"]["bounds"] = \
              serialise_bounds(rf_np, kwargs["minimizer_kwargs"]["bounds"])

        res = basinhopping(J, m_global, **kwargs)

    elif bounds != None:
        bounds = serialise_bounds(rf_np, bounds)
        res = scipy_minimize(J, m_global, method=method, bounds=bounds, **kwargs)
    else:
        res = scipy_minimize(J, m_global, method=method, **kwargs)

    rf_np.set_parameters(np.array(res["x"]))
    m = [p.data() for p in rf_np.parameter]
    return m

def minimize_custom(rf_np, bounds=None, **kwargs):
    ''' Interface to the user-provided minimisation method '''

    try:
        algo = kwargs["algorithm"]
        del kwargs["algorithm"]
    except KeyError:
        raise KeyError, 'When using a "Custom" optimisation method, you must pass the optimisation function as the "algorithm" parameter. Make sure that this function accepts the same arguments as scipy.optimize.minimize.' 

    m = [p.data() for p in rf_np.parameter]
    m_global = rf_np.obj_to_array(m)
    J = rf_np.__call__
    dJ = lambda m: rf_np.derivative(m, taylor_test=dolfin.parameters["optimization"]["test_gradient"], 
                                       seed=dolfin.parameters["optimization"]["test_gradient_seed"],
                                       forget=None)
    H = rf_np.hessian

    if bounds != None:
        bounds = serialise_bounds(rf_np, bounds)

    res = algo(J, m_global, dJ, H, bounds, **kwargs)

    try:
        rf_np.set_parameters(np.array(res))
    except Exception as e:
        raise e, "Failed to updated the optimised parameter value. Are you sure your custom optimisation algorithm returns an array containing the optimised values?" 
    m = [p.data() for p in rf_np.parameter]
    return m

optimization_algorithms_dict = {'L-BFGS-B': ('The L-BFGS-B implementation in scipy.', minimize_scipy_generic),
                                'SLSQP': ('The SLSQP implementation in scipy.', minimize_scipy_generic),
                                'TNC': ('The truncated Newton algorithm implemented in scipy.', minimize_scipy_generic), 
                                'CG': ('The nonlinear conjugate gradient algorithm implemented in scipy.', minimize_scipy_generic), 
                                'BFGS': ('The BFGS implementation in scipy.', minimize_scipy_generic), 
                                'Nelder-Mead': ('Gradient-free Simplex algorithm.', minimize_scipy_generic),
                                'Powell': ('Gradient-free Powells method', minimize_scipy_generic),
                                'Newton-CG': ('Newton-CG method', minimize_scipy_generic),
                                'Anneal': ('Gradient-free simulated annealing', minimize_scipy_generic),
                                'basinhopping': ('Global basin hopping method', minimize_scipy_generic),
                                'COBYLA': ('Gradient-free constrained optimization by linear approxition method', minimize_scipy_generic),
                                'Custom': ('User-provided optimization algorithm', minimize_custom)
                                }

def print_optimization_methods():
    ''' Prints the available optimization methods '''

    print 'Available optimization methods:'
    for function_name, (description, func) in optimization_algorithms_dict.iteritems():
        print function_name, ': ', description

def minimize(rf, method='L-BFGS-B', scale=1.0, in_euclidian_space=False, **kwargs):
    ''' Solves the minimisation problem with PDE constraint:

           min_m func(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, func is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimization problem is solved using a gradient based optimization algorithm and the functional gradients are computed by solving the associated adjoint system.

        The function arguments are as follows:
        * 'rf' must be a ReducedFunctional object. 
        * 'method' specifies the optimization method to be used to solve the problem. The available methods can be listed with the print_optimization_methods function.
        * 'scale' is a factor to scale to problem (default: 1.0). 
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        * 'in_euclidian_space' specifies if problem should be internally transformed to the Euclidian inner product. Since most optimisation implementations assume 
          the Euclidian inner product, this is usually a good choice.
        
        Additional arguments specific for the optimization algorithms can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization algorithm. For detailed information about which arguments are supported for each optimization algorithm, please refer to the documentaton of the optimization algorithm.
        '''

    if isinstance(rf, ReducedFunctional):
        rf_np = ReducedFunctionalNumPy(rf, in_euclidian_space)
    else:
        rf_np = rf # Assume the user knows what he is doing - he might for example written his own reduced functional class (as in OpenTidalFarm)

    rf_np.scale = scale

    try:
        algorithm = optimization_algorithms_dict[method][1]
    except KeyError:
        raise KeyError, 'Unknown optimization method ' + method + '. Use print_optimization_methods() to get a list of the available methods.'

    if algorithm == minimize_scipy_generic:
        # For scipy's generic inteface we need to pass the optimisation method as a parameter. 
        kwargs["method"] = method 

    opt = algorithm(rf_np, **kwargs)

    if len(opt) == 1:
        return opt[0]
    else:
        return opt

def maximize(rf, method='L-BFGS-B', scale=1.0, in_euclidian_space=False, **kwargs):
    ''' Solves the maximisation problem with PDE constraint:

           max_m func(u, m) 
             s.t. 
           e(u, m) = 0
           lb <= m <= ub
           g(m) <= u
           
        where m is the control variable, u is the solution of the PDE system e(u, m) = 0, func is the functional of interest and lb, ub and g(m) constraints the control variables. 
        The optimization problem is solved using a gradient based optimization algorithm and the functional gradients are computed by solving the associated adjoint system.

        The function arguments are as follows:
        * 'rf' must be a ReducedFunctional object. 
        * 'method' specifies the optimization method to be used to solve the problem. The available methods can be listed with the print_optimization_methods function.
        * 'scale' is a factor to scale to problem (default: 1.0). 
        * 'bounds' is an optional keyword parameter to support control constraints: bounds = (lb, ub). lb and ub must be of the same type than the parameters m. 
        
        Additional arguments specific for the optimization methods can be added to the minimize functions (e.g. iprint = 2). These arguments will be passed to the underlying optimization method. For detailed information about which arguments are supported for each optimization method, please refer to the documentaton of the optimization algorithm.
        '''
    return minimize(rf, method, scale=-scale, in_euclidian_space=in_euclidian_space, **kwargs)

minimise = minimize
maximise = maximize
