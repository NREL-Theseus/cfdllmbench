import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
n = 3.0  # polytropic index
R0 = 5.0  # scaling parameter

# Discretization 
Nr = 200  # radial grid points
r = np.linspace(0, 1, Nr)
dr = r[1] - r[0]

# Initial guess 
f = R0**((2)/(n-1)) * (1 - r**2)**2

# Finite difference matrix setup
main_diag = np.zeros(Nr)
lower_diag = np.zeros(Nr-1)
upper_diag = np.zeros(Nr-1)

# Interior points finite difference discretization
for i in range(1, Nr-1):
    # Radial derivative terms with variable coefficient
    main_diag[i] = -2 - (2/r[i]) * (1 if r[i] != 0 else 0)
    lower_diag[i-1] = 1 + (1/r[i])
    upper_diag[i] = 1 - (1/r[i])

# Boundary conditions
main_diag[0] = 1  # regularity at center
f[0] = 0
main_diag[-1] = 1  # Dirichlet at outer radius 
f[-1] = 0

# Construct sparse matrix
diagonals = [main_diag, lower_diag, upper_diag]
offsets = [0, -1, 1]
A = sp.diags(diagonals, offsets, shape=(Nr, Nr))

# Nonlinear solve via Newton iteration
max_iter = 100
tol = 1e-8

for _ in range(max_iter):
    # Nonlinear source term 
    source = f**n
    
    # Residual 
    res = A.dot(f) + source
    
    # Jacobian 
    J = A + sp.diags(n * f**(n-1), 0, shape=(Nr,Nr))
    
    # Newton update
    df = spla.spsolve(J, -res)
    f += df
    
    # Convergence check
    if np.max(np.abs(df)) < tol:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/f_Lane_Emden_Equation.npy', f)