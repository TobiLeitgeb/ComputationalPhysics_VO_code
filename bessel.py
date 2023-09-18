import numpy as np
import numba as nb
import scipy.special as sp
import matplotlib.pyplot as plt




@nb.jit(nopython=True)
def upward_recursion(l,x):
    "upward recursion. only stabel for x>l --> roundoff errors accumulate for x<l"
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
    return J

@nb.jit(nopython=True)
def J_lx(l,x):
    "spherical bessel function of the first kind. uses upward recursion for x>l and downward recursion for x<l"
    assert l >= 0, "l must be >= 0"
    # check that x is not to close to zero otherwose we get problems later
    if x < 0.06 and l > 0:
        return 0
    elif x < 0.06 and l == 0:
        return 1
    # check which recursion to use
    if x < l:
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
            x_start = (boundary_points[n] + boundary_points[n+1])/2 #+ 0.2
            roots[l,n] = newton_raphson(l,x_start)

        roots[l,-1] = roots[l,-2] + np.pi   # we need to add a "fake" root to the end of the array to define the boundary for the last root n = n_max this is not a real root but will not be returned
        boundary_points = roots[l].copy()
    return roots[:,:-1]


# =====================================================================================================
# ======================================== PLOTS ======================================================
# =====================================================================================================


def plot_wavefunctions(r, np_eigvals, np_eigvecs, k_il, n_roots, R, N, squared=False, symmetric=False):
    "plots the wavefunctions of the first N states"

    r = np.linspace(0,1.5*R,1000)
    r_notsymm = r
    r_symm = np.concatenate((-r[::-1], r[1:]))

    N = 9
    indices_flat = np.argpartition(np_eigvals.flatten(), N)[:N]
    sorted_indices = indices_flat[np.argsort(np_eigvals.flatten()[indices_flat])]
    indices_tuples = np.unravel_index(sorted_indices, np_eigvals.shape)

    fig, ax = plt.subplots(int(np.ceil(N/3)),3,figsize = (22,5*int(np.ceil(N/3))))
    if squared:
        fig.suptitle(f"Probability density of the first {N} states " + "$|\\Psi|^2$", fontsize = 18)
    else:
        fig.suptitle(f"Wave function of the first {N} states " + "$\\Psi$", fontsize = 18)
    fig.set_facecolor('white')
    for n, (row,column) in enumerate(zip(*indices_tuples)):
        r = r_notsymm
        l = row
        state = np_eigvecs[row,:,column]
        wavefunction = np.zeros(len(r))
        ax = ax.flatten()
        for i in range(n_roots):
            alpha = calculate_alpha(i,l,R, k_il)
            # spherical = sp.spherical_jn(l,k_il[i,l] * r)
            spherical = np.vectorize(J_lx)(l,k_il[i,l] * r)
            basis_func = spherical * alpha  # * spherical harmonics, which do not depend on r
            wavefunction += basis_func * state[i]
        if symmetric:
            wavefunction = np.concatenate((wavefunction[::-1], wavefunction[1:]))
            r = r_symm
        if squared:
            wavefunction = np.abs(wavefunction)**2
        ax[n].plot(r, wavefunction)
        ax[n].grid(alpha=0.5)
        ax[n].set_title(f"state = {n}, l = {row}, i = {column}, eigenvalue = " + "{:.4e}".format(np_eigvals[row,column]))


# Other than in the main.ipynb funciton, this calculate_alpha function also takes the radius R and k_il as argument
@nb.jit(nopython=True)
def calculate_alpha(j,l,R, k_il):
    if l == 0:
        return np.sqrt(2/R**3) * (j+1) * np.pi
    else:
        return 1 / J_lx(l-1,k_il[j,l] * R) * np.sqrt(2/R**3)