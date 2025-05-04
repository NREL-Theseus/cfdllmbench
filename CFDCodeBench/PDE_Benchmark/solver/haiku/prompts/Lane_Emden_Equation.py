import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid setup
N = 1000  # Number of grid points
dr = 1.0/N
r = np.linspace(0, 1, N+1)  # Include r=0
r = r[1:]  # Remove r=0 point since equation is singular there

# Initial guess
R0 = 5.0
n = 3.0
f = R0**(2/(n-1)) * (1 - r**2)**2

# Setup sparse matrix for Laplacian operator
# d^2f/dr^2 + (2/r)df/dr using central differences
main_diag = -2.0 * np.ones(N-1)
upper_diag = np.ones(N-2)
lower_diag = np.ones(N-2)

# Add (2/r)df/dr terms
for i in range(N-1):
    if i > 0:
        lower_diag[i-1] += 1.0/r[i]
    if i < N-2:
        upper_diag[i] -= 1.0/r[i]

A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
A = A/(dr**2)

# Newton iteration
max_iter = 100
tol = 1e-10
for iter in range(max_iter):
    # Compute residual
    f_laplace = np.zeros(N-1)
    f_laplace[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2])/(dr**2) + \
                      (2/r[1:-1])*(f[2:] - f[:-2])/(2*dr)
    f_laplace[0] = (f[1] - 2*f[0] + f[0])/(dr**2) + \
                   (2/r[0])*(f[1] - f[0])/(dr)
    
    R = f_laplace + f**n
    
    # Compute Jacobian
    J = A.copy()
    J = J.tolil()
    J.setdiag(J.diagonal() + n*f**(n-1))
    J = J.tocsr()
    
    # Solve for update
    df = spsolve(J, -R)
    
    # Update solution
    f = f + df
    
    # Check convergence
    if np.max(np.abs(df)) < tol:
        break

# Add r=0 point back with value determined by symmetry
f_final = np.concatenate(([f[0]], f))
r_final = np.concatenate(([0], r))

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/f_final_Lane_Emden_Equation.npy', f_final)