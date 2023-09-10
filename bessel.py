import numpy as np
import numba as nb
import scipy.special as sp

#@nb.jit(nopython=True)
def downward_recursion2(x):
    M = 100
    J = np.zeros(M+1)
    J[M] = 0
    J[M-1] = 1
    for l in range(M-1,0,-1):
        J[l-1] = (2*l+1)/x*J[l] - J[l+1]
    const = (np.sin(x)/x)/J[0]
    J = J*const
    return J

#@nb.jit(nopython=True)
def J_lx(l,x):
    "not very efficient, but works"
    J = downward_recursion2(x)
    return J[l]

def dJ_lx(l,x):
    "not very efficient, but works"
    J = downward_recursion2(x)
    return J[l-1] - (l+1)/x*J[l]

@nb.jit(nopython=True)   
def step(l,x):
    "a step for the newton raphson method"
    J = downward_recursion2(x)
    return J[l]/(l/x*J[l] - J[l+1])

def newton_raphson_scratch(l,x_0,max_steps=5000):
    x = x_0
    for s in range(max_steps):
        x_old = x
        #x = x_old - J_lx(l,x)/dJ_lx(l,x)
        #x = x_old - sp.spherical_jn(l,x)/sp.spherical_jn(l,x,derivative=True)
        #x = x_old - downward_recursion2(x)[l]/derivative_J(x)[l]
        x = x_old - step(l,x_old)
        if np.abs(x-x_old) < 1e-10 and x < x_0:
            return x
        
    pass

def newton_raphson_scipy(l,x_0,max_steps=5000):
    x = x_0
    for s in range(max_steps):
        x_old = x
        x = x_old - sp.spherical_jn(l,x)/sp.spherical_jn(l,x,derivative=True)
        if np.abs(x-x_old) < 1e-10 and x < x_0:
            return x
        
    pass
    

def bessel_roots(l_max, n_roots,scipy=False):
    if scipy:
        newton_raphson = newton_raphson_scipy
    else:
        newton_raphson = newton_raphson_scratch
    roots = np.zeros((l_max+1,n_roots+1))
    x_start = 1
    n = 0
    # for l = 0 we have J_0(x) = sin(x)/x so we know the roots are at x = n*pi
    #we know that roots for J_l(x) are between the two roots of J_{l-1}(x)
    roots[0] = np.arange(1,n_roots+2)*np.pi 
    boundary_points = roots[0].copy()
    for l in range(1,l_max+1):
        for n in range(n_roots):
            x_start = (boundary_points[n]+boundary_points[n+1])/2 
            roots[l,n] = newton_raphson(l,x_start)

        roots[l,-1] = roots[l,-2] + np.pi * 1.3  # we need to add a "fake" root to the end of the array to define the boundary this is not a real root but will not be returned
        boundary_points = roots[l].copy()
    return roots[:,:-1]