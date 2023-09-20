import numpy as np
import matplotlib.pyplot as plt

def jacobi_diagonalization(A: np.ndarray, tol=1e-8, max_iter=100):
    """Jacobi diagonalization routine for a symmetric matrix.
    Args:
        A (nd.array): Matrix to be diagonalized.
        tol (float): tolerance. Defaults to 1e-6.
        max_iter (int): max iterations. Defaults to 100.

    Returns:
        tuple: (list of eigenvalues, matrix of eigenvectors)
    """
    n = A.shape[0]
    P = np.eye(n)  # Initialize the orthogonal transformation matrix P

    for _ in range(max_iter):
        # Find the maximum off-diagonal element in A
        max_off_diag = 0.0
        max_i, max_j = 0, 0
        for i in range(n):
            for j in range(i + 1, n): # Only check the upper triangular part of A
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = np.abs(A[i, j])
                    max_i, max_j = i, j

        # Check if the maximum off-diagonal element is below the tolerance
        if max_off_diag < tol:
            break

        # Compute the rotation angle
        if A[max_i, max_i] == A[max_j, max_j]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[max_i, max_j] / (A[max_i, max_i] - A[max_j, max_j]))

        # Compute elements of the rotation matrix
        c = np.cos(theta)
        s = np.sin(theta)

        # Construct the rotation matrix
        R = np.eye(n)
        R[max_i, max_i] = c
        R[max_j, max_j] = c
        R[max_i, max_j] = -s
        R[max_j, max_i] = s

        # Update A and P using the rotation
        A = np.dot(np.dot(R.T, A), R)
        P = np.dot(P, R)

    eigenvalues = np.diag(A)
    eigenvectors = P

    return eigenvalues, eigenvectors




class Jacobi:
    "Translated code of the jacobi routine from Numerical Recipes in C++."
    def __init__(self, aa):
        self.n = aa.shape[0]
        self.a = aa.copy()
        self.v = np.eye(self.n)
        self.d = np.zeros(self.n)
        self.nrot = 0
        self.EPS = np.finfo(float).eps

    def rotate(self, a, s, tau, i, j, k, l):
        g = a[i, j]
        h = a[k, l]
        a[i, j] = g - s * (h + g * tau)
        a[k, l] = h + s * (g - h * tau)

    def jacobi_diagonalization(self):
        n = self.n
        a = self.a
        v = self.v
        d = self.d
        nrot = self.nrot

        b = np.zeros(n)
        z = np.zeros(n)

        for ip in range(n):
            for iq in range(n):
                v[ip, iq] = 0.0
            v[ip, ip] = 1.0

        for ip in range(n):
            b[ip] = d[ip] = a[ip, ip]
            z[ip] = 0.0

        for i in range(50):
            sm = 0.0
            for ip in range(n - 1):
                for iq in range(ip + 1, n):
                    sm += abs(a[ip, iq])

            if sm == 0.0:
                eigenvalues, eigenvectors = np.linalg.eigh(a)
                idx = np.argsort(eigenvalues)[::-1]
                self.d = eigenvalues[idx]
                self.v = eigenvectors[:, idx]
                return

            if i < 4:
                tresh = 0.2 * sm / (n * n)
            else:
                tresh = 0.0

            for ip in range(n - 1):
                for iq in range(ip + 1, n):
                    g = 100.0 * abs(a[ip, iq])

                    if i > 4 and g <= self.EPS * abs(d[ip]) and g <= self.EPS * abs(d[iq]):
                        a[ip, iq] = 0.0
                    elif abs(a[ip, iq]) > tresh:
                        h = d[iq] - d[ip]

                        if g <= self.EPS * abs(h):
                            t = a[ip, iq] / h
                        else:
                            theta = 0.5 * h / a[ip, iq]
                            t = 1.0 / (abs(theta) + np.sqrt(1.0 + theta * theta))
                            if theta < 0.0:
                                t = -t
                        c = 1.0 / np.sqrt(1 + t * t)
                        s = t * c
                        tau = s / (1.0 + c)
                        h = t * a[ip, iq]
                        z[ip] -= h
                        z[iq] += h
                        d[ip] -= h
                        d[iq] += h
                        a[ip, iq] = 0.0

                        for j in range(ip):
                            self.rotate(a, s, tau, j, ip, j, iq)
                        for j in range(ip + 1, iq):
                            self.rotate(a, s, tau, ip, j, j, iq)
                        for j in range(iq + 1, n):
                            self.rotate(a, s, tau, ip, j, iq, j)
                        for j in range(n):
                            self.rotate(v, s, tau, j, ip, j, iq)
                        nrot += 1

            for ip in range(n):
                b[ip] += z[ip]
                d[ip] = b[ip]
                z[ip] = 0.0

        raise Exception("Too many iterations in routine jacobi")
