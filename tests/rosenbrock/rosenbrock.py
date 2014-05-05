from moola import *
import pytest
class MyFunctional(Functional):
    def __call__(self, val):
        x, y = val.data
        r = (1 - x)**2 + 100*(y - x**2)**2
        events.increment("Functional evaluation")
        return r

    def derivative(self, val):
        x, y = val.data
        #print "current x =", x, y 

        dx = - 2*(1 - x) + 100*2*(y - x**2) * (-2*x)
        dy = 100 * 2 * (y - x**2)
        dr = (dx, dy)

        events.increment("Derivative evaluation")
        return NumpyDualVector(dr)

    def hessian(self, val):
        x, y = val.data
        def hes(vec):
            v = vec.data
            dxx = 2. + 1200.*x**2 - 400.*y
            dxy = -400.*x
            dyy = 200.
            d2v = (dxx * v[0] + dxy * v[1], dxy*v[0] +dyy*v[1] )
            events.increment("Hessian evaluation")
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


if __name__ == '__main__':
    prob, x0, opt = moola_problem()
    opt.update({"gtol": 1e-12, 'mem_lim': 2, 'initial_bfgs_iterations': 1, 'ncg_hesstol': 0})
    solver = HybridCG(prob, x0, options = opt)
    sol = solver.solve()
    
