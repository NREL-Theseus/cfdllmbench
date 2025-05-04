import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 10.0  # Domain length
T = 10.0  # Total simulation time
a = 1e-4  # Diffusion coefficient 
b = 2e-4  # Dispersion coefficient
n = 20    # Initial condition parameter

# Discretization
Nx = 200  # Spatial points 
Nt = 1000  # Time steps
dx = L / (Nx - 1)
dt = T / Nt
x = np.linspace(0, L, Nx)

# Initial condition 
def initial_condition(x):
    return 0.5/n * np.log(1 + np.cosh(n)**2 / np.cosh(n*(x - 0.2*L))**2)

u = initial_condition(x)

# Time integration using finite difference method
for _ in range(Nt):
    # Compute derivatives using central differences
    u_x = np.gradient(u, dx)
    u_xx = np.gradient(u_x, dx)
    u_xxx = np.gradient(u_xx, dx)
    
    # KdV-Burgers equation update 
    u_new = u - dt * (u * u_x) + dt * a * u_xx + dt * b * u_xxx
    
    # Enforce periodic boundary conditions
    u_new[0] = u_new[-1]
    
    u = u_new

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_1D_KdV_Burgers_Equation.npy', u)