import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
Re_tau = 395
kappa = 0.42
A = 25.4
mu = 1 / Re_tau

# Discretization parameters
ny = 200
y = np.linspace(0, 2, ny)
dy = y[1] - y[0]

# Initialize solution arrays
u = np.zeros(ny)
mu_eff = np.zeros(ny)

# Compute y+
y_plus = y * Re_tau

# Compute effective viscosity using Cess model
def compute_mu_eff(y):
    mu_t = mu * (0.5 * (1 + 1/9 * kappa**2 * Re_tau**2 * 
                        (2*y - y**2)**2 * 
                        (3 - 4*y + 2*y**2)**2 * 
                        (1 - np.exp(-y_plus/A))**2)**0.5 - 0.5)
    return mu + mu_t

# Solve using finite difference method
def solve_momentum():
    # Construct matrix A and vector b
    diags = np.zeros((3, ny))
    
    # Interior points
    for i in range(1, ny-1):
        mu_eff_i = compute_mu_eff(y[i])
        mu_eff_im = compute_mu_eff(y[i-1])
        mu_eff_ip = compute_mu_eff(y[i+1])
        
        # Coefficients for finite difference discretization
        a_im = mu_eff_im / dy**2
        a_i = -(mu_eff_im + mu_eff_ip) / dy**2
        a_ip = mu_eff_ip / dy**2
        
        diags[0, i] = a_im
        diags[1, i] = a_i
        diags[2, i] = a_ip
    
    # Boundary conditions
    diags[1, 0] = 1.0  # u(0) = 0
    diags[1, -1] = 1.0  # u(2) = 0
    
    # Source term
    b = np.ones(ny)
    b[0] = 0.0
    b[-1] = 0.0
    
    # Create sparse matrix
    A = sp.diags([diags[0, 1:], diags[1, :], diags[2, :-1]], 
                 offsets=[-1, 0, 1], shape=(ny, ny))
    
    # Solve linear system
    u = spla.spsolve(A, b)
    
    return u

# Solve problem
u = solve_momentum()

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)