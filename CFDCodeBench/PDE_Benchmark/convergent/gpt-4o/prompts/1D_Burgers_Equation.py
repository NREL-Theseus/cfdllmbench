import numpy as np

# Parameters
nu = 0.07
L = 2 * np.pi
T = 0.14 * np.pi
nx = 400
nt = 4000
dx = L / (nx - 1)
dt = T / nt

# Spatial and temporal grids
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
u = -2 * nu / phi * np.gradient(phi, dx) + 4

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:-1] = (un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) +
               nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
    
    # Periodic boundary conditions
    u[0] = (un[0] - un[0] * dt / dx * (un[0] - un[-2]) +
            nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2]))
    u[-1] = u[0]

# Save the final solution
np.save('/home/weichao/Downloads/Code_Generation_Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_1D_Burgers_Equation.npy', u)