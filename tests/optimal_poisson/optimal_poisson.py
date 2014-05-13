from dolfin import *
from dolfin_adjoint import *
import pytest
import moola


dolfin.set_log_level(ERROR)
parameters['std_out_all_processes'] = False

def solve_pde(u, V, m):
    v = TestFunction(V)
    w = TrialFunction(V)
    a = (.5 * w*v + (4*pi**2)**-1 * inner(grad(w), grad(v)))*dx
    L = m*v * dx
    #bc = DirichletBC(V, 0.0, "on_boundary")
    solve(a == L, u)

def randomly_refine(initial_mesh, ratio_to_refine= .3):
    import numpy.random
    numpy.random.seed(0)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cell_markers = cf)
    
    

@pytest.fixture(params=[("structured mesh", 16), ("nonstructured mesh", 3)])
def moola_problem(request):
    if request.param[0] == "structured mesh":
        N = request.param[1]
        mesh = UnitSquareMesh(N,N)
    if request.param[0] == "nonstructured mesh":
        number_of_refinements = request.param[1]
        mesh = Mesh(Rectangle(0,0,1,1), 10)
        for k in xrange(number_of_refinements):
            mesh = randomly_refine(mesh)
    V = FunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, "CG", 2)
    
    u = Function(V, name='State')
    expr = Expression('2*x[0]-1.')

    bc = DirichletBC(Q, 0.0, "on_boundary")
    m = interpolate(expr, Q)
    
    alpha = Constant(0.1)
    
    u_d = interpolate(Expression('pow(sin(pi*x[0])*sin(pi*x[1]),2)'), Q)
    #from numpy.random import rand
    #epsilon = 0.05
    #u_d.vector()[:] += epsilon * (rand(len(u_d.vector()))*2 - 1)
    #u_d = Function(V)
    #solve_pde(u_d, V, m_ex)
    
    J = Functional((.5*inner(u-u_d, u-u_d))*dx+.5*alpha*m**2*dx)

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation 
    rf = ReducedFunctional(J, InitialConditionParameter(m, value=m))
    
    x_init = moola.DolfinPrimalVector(m)
    options = {'jtol': None, 'gtol': None, 'display': 3, 'maxiter' : 100,}
    return rf.moola_problem(), x_init, options

@pytest.mark.parametrize("bfgs_options,bfgs_expected",
                         [ ({"gtol": 1e-4, 'mem_lim':30}, [11]),
                           ({"jtol": 1e-7, 'mem_lim':30}, [12]),])
def test_LBFGS(bfgs_options, bfgs_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(bfgs_options)
    solver = moola.BFGS(problem, x_init, options = options)
    sol = solver.solve()
    assert len(sol['lbfgs']) == min(options['mem_lim'], sol['iteration'])
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] in bfgs_expected

@pytest.mark.parametrize("nlcg_options,nlcg_expected",
                         [
                           ({"gtol": 1e-3, "beta_rule": "polak-ribiere-polyak"}, [14]),
                           ({"jtol": 1e-5, "beta_rule": "polak-ribiere-polyak"}, [11]),
                           ({"gtol": 1e-3, "beta_rule": "hager-zhang"}, [17]),
                           ({"jtol": 1e-5, "beta_rule": "hager-zhang"}, [9]),
                           #({"gtol": 1e-7, "beta_rule": "hestenes-stiefel"}, 33),
                           #({"jtol": 1e-12, "beta_rule": "hestenes-stiefel"}, 32)
                           ])                           
def test_NonlinearCG(nlcg_options, nlcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(nlcg_options)
    solver = moola.NonLinearCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] in nlcg_expected


@pytest.mark.parametrize("newtcg_options,newtcg_expected",
                         [ ({"gtol": 1e-8, 'ncg_hesstol': 0}, 6),
                           ({"jtol": 1e-9, 'ncg_hesstol': 0}, 6)])
def test_NewtonCG(newtcg_options, newtcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(newtcg_options)
    solver = moola.NewtonCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == newtcg_expected

'''

def compute_Ax(x, V):
    u = Function(V)
    #embed()
    u.assign(x.data)
    v = TestFunction(V)
    #u, v = TrilFunction, TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    Ax =  assemble(a, bcs=bc)
    v = Function(V, Ax)
    return moola.DolfinDualVector(v)

def compute_A(V):
    u, v = TrialFunction(V), TestFunction(V)
    a = (inner(grad(u), grad(v)) +u*v) * dx
    A = assemble(a)
    return A

def compute_M(V):
    u, v = TrialFunction(V), TestFunction(V)
    a = u*v * dx
    A = assemble(a)
    return A

def compute_Mx(x, V):
    u = Function(V)
    #embed()
    u.assign(x.data)
    v = TestFunction(V)
    #u, v = TrilFunction, TestFunction(V)
    a = u*v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    Mx =  assemble(a, bcs=bc)
    v = Function(V, Mx)
    return moola.DolfinDualVector(v)

def compute_Ainvx(x, V):
    u = Function(V)
    f = x.data
    solve_pde(u, V, f)
    return moola.DolfinPrimalVector(u)





def compute_errors(u, m):
    solve_pde(u, V, m)

    

    # Define the analytical expressions
    m_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")
    u_analytic = Expression("1/(2*pi*pi)*sin(pi*x[0])*sin(pi*x[1])")

    # Compute the error
    control_error = errornorm(m_analytic, m)
    state_error = errornorm(u_analytic, u)

    print "Error in state: {}.".format(state_error)
    print "Error in control: {}.".format(control_error)

    return control_error, state_error


if __name__ == "__main__":

    n = 16
    mesh = UnitSquareMesh(n, n)

    def ref(mesh):
        cf = CellFunction("bool", mesh)
        subdomain = CompiledSubDomain('x[0]>.5')
        subdomain.mark(cf, True)
        return refine(mesh, cf)

    #mesh = ref(mesh)
    #mesh = ref(mesh)

    #plot(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name='State')
    W = FunctionSpace(mesh, "CG", 1)
    m = Function(W, name='Control')
    alpha = Constant(1e-5)
    u_d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) 

    J = Functional((.5*inner(u-u_d, u-u_d))*dx+.5*alpha*m**2*dx)

    # Run the forward model once to create the annotation
    solve_pde(u, V, m)

    # Run the optimisation 
    rf = ReducedFunctional(J, InitialConditionParameter(m, value=m))

    opt_package = "moola" # Choose from moola, scipy, ipopt

    if opt_package == "moola":
        try:
            import moola
        except ImportError:
            import sys
            info_blue("moola bindings unavailable, skipping test")
            sys.exit(0)

        problem = rf.moola_problem()
        m_moola = moola.DolfinPrimalVector(m)
        
        def dual_to_primal(x):
                return x.primal()
        
        def precond_matvec(x):
            y1 = dual_to_primal(x)
            y2 = compute_Ax(y1,V)
            y3 = dual_to_primal(y2)
            y4 = compute_Ax(y3,V)
            y5 = dual_to_primal(y4)
            return y5


        A = compute_A(V)
        M = compute_M(V)

        def test(m):
            x = m.data.vector().copy()
            b = M * x.copy()
            solve(A, x, b)
            b =  M * x
            solve(A, x, b)
            b =  M * x
            u = Function(V, b)
            return moola.DolfinDualVector(u)
        
       

        def test2(m):
            x = m.data.vector().copy()
            b = x.copy()
            solve(M, x, b)
            b =  A * x
            solve(M, x, b)
            b =  A * x
            solve(M, x, b)
            
            u = Function(V, x)
            return moola.DolfinPrimalVector(u)
            
        
        from moola.algorithms.bfgs import LinearOperator
        C = LinearOperator(dual_to_primal)
        B = LinearOperator(test2)
        H = LinearOperator(test)
        #embed()
        solver = moola.NewtonCG(problem, m_moola, precond = B,
                                options={'gtol': 1e-14, 'maxiter': 100, 
                                         'ncg_reltol':1e-10, 'ncg_hesstol': 1e-50, 'display':3})
        #solver = moola.BFGS(problem, m_moola,
        #                    options={'jtol':None, 'gtol': 1e-6, 'maxiter': 100, 'mem_lim': 10})

        #solver = moola.NonLinearCG(problem, m_moola,
        #                    options={'jtol':None, 'gtol': 1e-6,
        #                             'maxiter': 100, 'beta_rule': 'hager-zhang'})

        from moola.algorithms import *
        hybrid_options = {'display':3, 'jtol': None, 'gtol': 1e-8, 'maxiter': 1000, 'ncg_maxiter': 1000,
                  'mem_lim' : 4, 'ncg_hesstol': 1e-25, 'ncg_reltol': 0.5, "initial_bfgs_iterations" : 12}
        #solver = HybridCG(problem, m_moola, options=hybrid_options)
        sol = solver.solve()
        m_opt = sol['control'].data

        print moola.events
        from IPython import embed; embed()
        if sol["objective"] is not None:
            assert abs(sol["objective"]) < 1e-9
        assert sol["iteration"] == 1

    elif opt_package == "scipy":
        #m_opt = minimize(rf, method="Newton-CG", options={"xtol": 1e-100})
        m_opt = minimize(rf, tol=1e-100, method="L-BFGS-B", options={"xtol": 1e-100, "maxiter": 100})

    elif opt_package == "ipopt":
        import pyipopt

        rfn  = ReducedFunctionalNumPy(rf)
        nlp = rfn.pyipopt_problem()
        m_opt = nlp.solve(full=False)


    #plot(m_opt, interactive=True)
    ctrl_err, state_err = compute_errors(u, m_opt)
    assert ctrl_err < 3e-2
    assert state_err < 1e-4



class MyFunctional(Functional):
    def __call__(self, val):
        x, y = val.data
        r = (1 - x)**2 + 100*(y - x**2)**2
        return r

    def derivative(self, val):
        x, y = val.data
        #print "current x =", x, y 

        dx = - 2*(1 - x) + 100*2*(y - x**2) * (-2*x)
        dy = 100 * 2 * (y - x**2)
        dr = (dx, dy)

        return NumpyDualVector(dr)

    def hessian(self, val):
        x, y = val.data
        def hes(vec):
            v = vec.data
            dxx = 2. + 1200.*x**2 - 400.*y
            dxy = -400.*x
            dyy = 200.
            d2v = (dxx * v[0] + dxy * v[1], dxy*v[0] +dyy*v[1] )
            return NumpyDualVector(d2v)
        return hes

@pytest.fixture
def moola_problem():
    objective = MyFunctional()
    x_init = NumpyPrimalVector((-3, -4))
    options = {'jtol': None, 'gtol': None, 'display': 3,
               'maxiter' : 1000,}# 'line_search_options' :{'xtol':1e-25, 'ignore_warnings':True}} 
    
    
    return Problem(objective), x_init, options

@pytest.mark.parametrize("bfgs_options,bfgs_expected",
                         [ ({"gtol": 1e-12, 'mem_lim': 2}, 19),
                           ({"jtol": 1e-12, 'mem_lim': 2}, 17),
                           ({"gtol": 1e-12, 'mem_lim': 100}, 30),
                           ({"jtol": 1e-12, 'mem_lim': 100}, 28)])
def test_LBFGS(bfgs_options, bfgs_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(bfgs_options)
    solver = BFGS(problem, x_init, options = options)
    sol = solver.solve()
    assert len(sol['lbfgs']) == min(options['mem_lim'], sol['iteration'])
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == bfgs_expected


@pytest.mark.parametrize("newtcg_options,newtcg_expected",
                         [ ({"gtol": 1e-12, 'ncg_hesstol':0}, 32),
                           ({"jtol": 1e-12, 'ncg_hesstol':0}, 32),
                           ({"gtol": 1e-12, 'ncg_hesstol':0, 'ncg_reltol':1e-20}, 31),
                           ({"jtol": 1e-12, 'ncg_hesstol':0, 'ncg_reltol':1e-20}, 31), ])
def test_NewtonCG(newtcg_options, newtcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(newtcg_options)
    solver = NewtonCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == newtcg_expected


@pytest.mark.parametrize("nlcg_options,nlcg_expected",
                         [ ({"gtol": 1e-12, "beta_rule": "fletcher-reeves"}, 536),
                           ({"jtol": 1e-12, "beta_rule": "fletcher-reeves"}, 119),
                           ({"gtol": 1e-12, "beta_rule": "polak-ribiere-polyak"}, 28),
                           ({"jtol": 1e-12, "beta_rule": "polak-ribiere-polyak"}, 25),
                           ({"gtol": 1e-12, "beta_rule": "hager-zhang"}, 37),
                           ({"jtol": 1e-12, "beta_rule": "hager-zhang"}, 35),
                           ({"gtol": 1e-12, "beta_rule": "liu-storey"}, 26),
                           ({"jtol": 1e-12, "beta_rule": "liu-storey"}, 23),
                           ({"gtol": 1e-12, "beta_rule": "hestenes-stiefel"}, 33),
                           ({"jtol": 1e-12, "beta_rule": "hestenes-stiefel"}, 32),
                           ({"gtol": 1e-12, "beta_rule": "dai-yuan"},109),
                           ({"jtol": 1e-12, "beta_rule": "dai-yuan"}, 55),
                           ({"gtol": 1e-12, "beta_rule": "conjugate_descent"},157),
                           ({"jtol": 1e-12, "beta_rule": "conjugate_descent"}, 63),
                           ({"gtol": 1e-02, "beta_rule": "steepest_descent"}, 647),
                           ({"jtol": 1e-06, "beta_rule": "steepest_descent"}, 295)])
def test_NonlinearCG(nlcg_options, nlcg_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(nlcg_options)
    solver = NonLinearCG(problem, x_init, options = options)
    sol = solver.solve()
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == nlcg_expected



@pytest.mark.parametrize("hybrid_options,hybrid_expected",
                         [ ({"gtol": 1e-12, 'mem_lim': 0, 'ncg_hesstol': 0}, 32),
                           ({"jtol": 1e-12, 'mem_lim': 0, 'ncg_hesstol': 0}, 32),
                           ({"gtol": 1e-12, 'mem_lim': 100, 'initial_bfgs_iterations': 100}, 30),
                           ({"jtol": 1e-12, 'mem_lim': 100, 'initial_bfgs_iterations': 100}, 28),
                           ({"gtol": 1e-12, 'mem_lim': 2, 'initial_bfgs_iterations': 0, 'ncg_hesstol': 0}, 38),
                           ({"jtol": 1e-12, 'mem_lim': 2, 'initial_bfgs_iterations': 0, 'ncg_hesstol': 0}, 38),
                           ({"gtol": 1e-15, 'mem_lim': 100, 'initial_bfgs_iterations': 1, 'ncg_hesstol': 0, 'ncg_reltol':1e-20}, 14),
                           ({"jtol": 1e-15, 'mem_lim': 100, 'initial_bfgs_iterations': 1, 'ncg_hesstol': 0, 'ncg_reltol':1e-20}, 14), ])
def test_HybridCG(hybrid_options, hybrid_expected, moola_problem):
    problem, x_init, options = moola_problem
    options.update(hybrid_options)
    solver = HybridCG(problem, x_init, options = options)
    sol = solver.solve()
    print sol['iteration']
    if options['gtol'] is not None:
        assert sol['grad_norm'] < options['gtol']
    if options['jtol'] is not None:
        assert sol['delta_J'] < options['jtol']
    assert sol['iteration'] == hybrid_expected



    
    

    
     
    
'''

if __name__ == '__main__':
    '''
    class Req(object):
        #param = ("structured mesh", 32)
        param = ("nonstructured mesh", 3)
    req = Req()
    prob, x0, opt = moola_problem(req)
    
    #opt.update({"gtol": 1e-5, 'mem_lim': 200, 'maxiter': 20})
    #opt.update({"gtol": 1e-8, 'beta_rule': 'hager-zhang', 'maxiter': 5})
    opt.update({'gtol':1e-3, 'ncg_hesstol':1e-20, 'ncg_reltol': 1e-1})

    #solver = moola.BFGS(prob, x0, options = opt)
    #solver = moola.NonLinearCG(prob, x0, options = opt)
    solver = moola.NewtonCG(prob, x0, options = opt)
    
    
    
    #opt.update({"gtol": 1e-6, 'ncg_hesstol': 1e-20, 'maxiter': 100})
    #solver = moola.NewtonCG(prob, x0, options = opt)

    #opt.update({"jtol": 1e-8, 'mem_lim': 30, 'maxiter': 100})
    #solver = moola.BFGS(prob, x0, options = opt)
    
    
    sol = solver.solve()
    mh = sol['control'].data

    V = mh.function_space()
   
    plot(mh)
    V = FunctionSpace(mh.function_space().mesh(), 'CG', 5)
    u = Function(V)
    solve_pde(u, V, mh)
    plot(u)
    interactive()
    #plot(u, interactive = True, title = 'state')
    import pylab
    #pylab.plot(pylab.np.arange(len(solver.history['objective'])), pylab.np.log10(solver.history['objective']))
    #pylab.show()

    #pylab.plot(pylab.np.arange(len(solver.history['objective'])), pylab.np.log10(solver.history['grad_norm']))
    #pylab.show()
    '''
