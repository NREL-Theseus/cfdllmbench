import numpy as np

# Parameters
L = 10.0  # Domain length
T = 10.0  # Total time
Nx = 1000  # Number of spatial points
Nt = 10000  # Number of time steps
dx = L/Nx
dt = T/Nt
x = np.linspace(0, L, Nx)
a = 1e-4  # Diffusion coefficient
b = 2e-4  # Dispersion coefficient
n = 20    # Parameter for initial condition

# Initial condition
def initial_condition(x):
    return 0.5/n * np.log(1 + (np.cosh(n)**2)/(np.cosh(n*(x-0.2*L))**2))

# Initialize solution array
u = initial_condition(x)

# Time stepping
for t in range(Nt):
    # Periodic boundary handling
    u_m2 = np.roll(u, 2)
    u_m1 = np.roll(u, 1)
    u_p1 = np.roll(u, -1)
    u_p2 = np.roll(u, -2)
    
    # Spatial derivatives using central differences
    du_dx = (u_p1 - u_m1)/(2*dx)
    d2u_dx2 = (u_p1 - 2*u + u_m1)/dx**2
    d3u_dx3 = (-u_p2 + 2*u_p1 - 2*u_m1 + u_m2)/(2*dx**3)
    
    # Update solution
    u = u - dt*(u*du_dx - a*d2u_dx2 - b*d3u_dx3)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_KdV_Burgers_Equation.npy', u)