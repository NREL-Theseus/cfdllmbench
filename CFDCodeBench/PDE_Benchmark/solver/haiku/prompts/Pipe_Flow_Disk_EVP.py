import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

# Parameters
Nr = 100  # Number of radial points
m = 1     # Azimuthal wavenumber
kz = 1    # Axial wavenumber
Re = 1e4  # Reynolds number

# Grid
r = np.linspace(0, 1, Nr)
dr = r[1] - r[0]

# Background flow
w0 = 1 - r**2
dw0dr = -2*r

# Differential operators in r
# First derivative
D1_data = np.array([-1, 1]) / (2*dr)
D1_offsets = np.array([-1, 1])
D1 = sparse.diags(D1_data, D1_offsets, shape=(Nr, Nr))

# Second derivative 
D2_data = np.array([1, -2, 1]) / dr**2
D2_offsets = np.array([-1, 0, 1])
D2 = sparse.diags(D2_data, D2_offsets, shape=(Nr, Nr))

# Operators for cylindrical coordinates
I = sparse.eye(Nr)
R_inv = sparse.diags(1/r, 0)
R_inv2 = sparse.diags(1/r**2, 0)

# Build system matrix
def build_matrix():
    # Block matrices
    L11 = -1/Re * (D2 + R_inv@D1 - R_inv2 - kz**2*I) + \
          w0@D1@kz + s*I - m**2*R_inv2
    L12 = -1j*m*R_inv
    L13 = D1
    
    L21 = 1j*m*R_inv
    L22 = -1/Re * (D2 + R_inv@D1 - R_inv2 - kz**2*I) + \
          w0@D1@kz + s*I - m**2*R_inv2
    L23 = 1j*m*R_inv
    
    L31 = D1 + R_inv
    L32 = 1j*m*R_inv
    L33 = -1/Re * (D2 + R_inv@D1 - kz**2*I) + \
          w0@D1@kz + s*I + dw0dr@I

    # Combine blocks
    L = sparse.bmat([[L11, L12, L13],
                     [L21, L22, L23], 
                     [L31, L32, L33]])
    
    return L

# Apply boundary conditions
def apply_bc(L):
    n = 3*Nr
    # No-slip at r=1
    L[Nr-1,:] = 0
    L[Nr-1,Nr-1] = 1
    L[2*Nr-1,:] = 0 
    L[2*Nr-1,2*Nr-1] = 1
    L[3*Nr-1,:] = 0
    L[3*Nr-1,3*Nr-1] = 1
    
    # Regularity at r=0
    L[0,:] = 0
    L[0,0] = 1
    L[Nr,:] = 0
    L[Nr,Nr] = 1
    L[2*Nr,:] = 0
    L[2*Nr,2*Nr] = 1
    
    return L

# Solve eigenvalue problem
s = 0.1  # Initial guess
L = build_matrix()
L = apply_bc(L)

eigenvalues, eigenvectors = eigs(L, k=1, sigma=s)
s = eigenvalues[0]

# Extract solution components
ur = eigenvectors[:Nr,0]
uth = eigenvectors[Nr:2*Nr,0] 
w = eigenvectors[2*Nr:,0]

# Normalize
norm = np.max(np.abs(w))
ur = ur/norm
uth = uth/norm
w = w/norm

# Save solutions
np.save('ur.npy', np.abs(ur))
np.save('uth.npy', np.abs(uth))
np.save('w.npy', np.abs(w))