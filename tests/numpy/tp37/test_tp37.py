'''
Solves Schittkowski's TP37 Problem.

    min     -x1*x2*x3
    s.t.:   x1 + 2.*x2 + 2.*x3 - 72 <= 0
            - x1 - 2.*x2 - 2.*x3 <= 0
            0 <= xi <= 42,  i = 1,2,3
    
    f* = -3456 , x* = [24, 12, 12]
'''
def objfunc(x):
    
    f = -x[0]*x[1]*x[2]
    g = [0.0]*2
    g[0] = x[0] + 2.*x[1] + 2.*x[2] - 72.0
    g[1] = -x[0] - 2.*x[1] - 2.*x[2]
    
    fail = 0
    return f,g, fail
    
