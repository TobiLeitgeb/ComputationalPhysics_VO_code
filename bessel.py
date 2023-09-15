import numpy as np
import numba as nb
import scipy.special as sp



@nb.jit(nopython=True)
def upward_recursion(l,x):
    "upward recursion. only stabel for x>l"
    J_0 = np.sin(x)/x
    J_1 = np.sin(x)/(x**2) - np.cos(x)/x
    if l == 0:
        return J_0
    J_l = J_1
    J_lm1 = J_0
    
    for n in range(1,l+1):
        J_lp1 = (2*n+1)/x * J_l - J_lm1
        J_lm1 = J_l
        J_l = J_lp1
    
    return J_lm1

@nb.jit(nopython=True)
def miller_downward_recursion(l,x):
    "Miller algorithm for downward recursion of spherical bessel functions"
    cons = 5
    # M is the starting l for the downward recursion
    M = np.where(l<100,100,l+int(np.sqrt(cons*l))) # if l<100 then M=100 otherwise set M as described in numerical recipes p. 278
    J = np.zeros(M+1)
    J[M] = 0
    J[M-1] = 1
    for l in range(M-1,0,-1): #downward recursion
        J[l-1] = (2*l+1)/x*J[l] - J[l+1]
    
    J_0 = np.sin(x)/x
    const = J_0/J[0]
    J = J*const #renormalize
    J = np.nan_to_num(J)
    return J

@nb.jit(nopython=True)
def J_lx(l,x):
    "spherical bessel function of the first kind. uses upward recursion for x>l and downward recursion for x<l"
    assert l >= 0 
    # check that x is not to close to zero
    if x < 1e-4:
        return 0
    # check if we have to use the downward recursion
    if x < l+1:
        return miller_downward_recursion(l,x)[l]
    else:
        return upward_recursion(l,x)

@nb.jit(nopython=True)
def derivative_J(l,x):
    "derivative of spherical bessel function"
    return J_lx(l-1,x) - (l+1)/x*J_lx(l,x)

@nb.jit(nopython=True)   
def step(l,x):
    "a step for the newton raphson method J_l/dJ_l. computed like this to save computation time(2-calls instead of 3 to J_lx())"
    J_l = J_lx(l,x)
    return J_l/(J_lx(l-1,x) - (l+1)/x*J_l)

#--------------------------------------------
#functions for finding the roots of the bessel functions.
#-------------------------------------------

@nb.jit(nopython=True)
def newton_raphson_scratch(l,x_0,max_steps=5000):
    "newton raphson method for finding the roots of spherical bessel functions. Iterativly solves x = x - J_l(x)/dJ_l(x)"
    x = x_0
    for s in range(max_steps):
        x_old = x
        x = x_old - step(l,x)
        if np.abs(x-x_old) < 1e-10:
            return x
    else:
        raise ValueError("newton raphson did not converge")
        
    
#just for testing purposes
def newton_raphson_scipy(l,x_0,max_steps=5000):
    "same as newton_raphson_scratch but uses scipy.special.spherical_jn instead of J_lx. Used for testing purposes"
    x = x_0
    for s in range(max_steps):
        x_old = x
        x = x_old - sp.spherical_jn(l,x)/sp.spherical_jn(l,x,derivative=True)
        if np.abs(x-x_old) < 1e-10:
            return x
    else:
        raise ValueError("newton raphson did not converge")
    

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
    #we use the roots of the previous l to find the roots of the next l
    boundary_points = roots[0].copy()
    for l in range(1,l_max+1):
        for n in range(n_roots):
            #we start in the middle of the interval between the roots of J_{l-1}(x)
            x_start = (boundary_points[n] + boundary_points[n+1])/2 + 0.2
            roots[l,n] = newton_raphson(l,x_start)

        roots[l,-1] = roots[l,-2] + np.pi   # we need to add a "fake" root to the end of the array to define the boundary for the last root n = n_max this is not a real root but will not be returned
        boundary_points = roots[l].copy()
    return roots[:,:-1]