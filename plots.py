import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

def plot_wavefunctions(r, np_eigvals, np_eigvecs, k_il, n_roots, R, N):
    "plots the wavefunctions of the first N states"

    r = np.linspace(0,1.5*R,1000)

    N = 9
    indices_flat = np.argpartition(np_eigvals.flatten(), N)[:N]
    sorted_indices = indices_flat[np.argsort(np_eigvals.flatten()[indices_flat])]
    indices_tuples = np.unravel_index(sorted_indices, np_eigvals.shape)

    fig, ax = plt.subplots(int(np.ceil(N/3)),3,figsize = (22,5*int(np.ceil(N/3))))
    fig.set_facecolor('white')
    for n, (row,column) in enumerate(zip(*indices_tuples)):
        # row ~ l
        l = row
        state = np_eigvecs[row,:,column]
        wavefunction = np.zeros(len(r))
        ax = ax.flatten()
        for i in range(n_roots):
            alpha = calculate_alpha(i,l)
            # spherical = sp.spherical_jn(l,k_il[i,l] * r)
            spherical = np.vectorize(J_lx)(l,k_il[i,l] * r)
            basis_func = spherical * alpha  # * spherical harmonics, which do not depend on r
            wavefunction += basis_func * state[i]
        ax[n].plot(r, wavefunction)
        ax[n].grid(alpha=0.5)
        ax[n].set_title(f"state = {n}, l = {row}, i = {column}, eigenvalue = " + "{:.4e}".format(np_eigvals[row,column]))