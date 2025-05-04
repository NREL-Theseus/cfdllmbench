import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# Parameters
Re = 1e4
k_z = 1
s = 0  # Eigenvalue (to be solved for)
n_r = 100  # Number of radial points
n_phi = 100  # Number of azimuthal points
r = np.linspace(0, 1, n_r)
phi = np.linspace(0, 2 * np.pi, n_phi)
dr = r[1] - r[0]
dphi = phi[1] - phi[0]

# Background flow
w0 = 1 - r**2

# Discretization matrices
D2_r = diags([1, -2, 1], [-1, 0, 1], shape=(n_r, n_r)) / dr**2
D2_r = D2_r.toarray()
D2_r[0, 0] = D2_r[-1, -1] = 1  # Dirichlet BC at r=0 and r=1

# Azimuthal derivative (Fourier space)
modes = np.fft.fftfreq(n_phi, d=dphi/(2*np.pi))
D_phi = np.diag(1j * modes)

# Construct the linear operator for the eigenvalue problem
L = np.zeros((3 * n_r * n_phi, 3 * n_r * n_phi), dtype=complex)

# Fill the operator matrix L
for i in range(n_r):
    for j in range(n_phi):
        idx = i * n_phi + j
        # u equation
        L[idx, idx] = s + w0[i] * 1j * k_z - 1/Re * (D2_r[i, i] + k_z**2)
        L[idx, idx + n_r * n_phi] = 1j * modes[j]  # dp/dphi
        L[idx, idx + 2 * n_r * n_phi] = 1j * k_z  # dp/dz

        # v equation
        L[idx + n_r * n_phi, idx + n_r * n_phi] = s + w0[i] * 1j * k_z - 1/Re * (D2_r[i, i] + k_z**2)
        L[idx + n_r * n_phi, idx] = 1j * modes[j]  # dp/dphi
        L[idx + n_r * n_phi, idx + 2 * n_r * n_phi] = 1j * k_z  # dp/dz

        # w equation
        L[idx + 2 * n_r * n_phi, idx + 2 * n_r * n_phi] = s + w0[i] * 1j * k_z - 1/Re * (D2_r[i, i] + k_z**2)
        L[idx + 2 * n_r * n_phi, idx] = 1j * modes[j]  # du/dr
        L[idx + 2 * n_r * n_phi, idx + n_r * n_phi] = 1j * modes[j]  # dv/dphi

# Solve the eigenvalue problem
eigenvalues, eigenvectors = eigs(L, k=1, which='LR')

# Extract the eigenvector corresponding to the largest eigenvalue
eigenvector = eigenvectors[:, 0]

# Reshape the eigenvector to get u, v, w
u = eigenvector[:n_r * n_phi].reshape((n_r, n_phi))
v = eigenvector[n_r * n_phi:2 * n_r * n_phi].reshape((n_r, n_phi))
w = eigenvector[2 * n_r * n_phi:].reshape((n_r, n_phi))

# Save the results
np.save('u.npy', u.real)
np.save('v.npy', v.real)
np.save('w.npy', w.real)