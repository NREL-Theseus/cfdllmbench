import numpy as np

# Domain parameters
x_start, x_end = -5, 5
t_start, t_end = 0, 10
nx = 200  # Number of spatial points
nt = 1000  # Number of time steps
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / nt

# Grid
x = np.linspace(x_start, x_end, nx)

# Parameters
c = 1.0  # Convection speed
epsilon = 5e-4  # Damping factor

# Initial condition
u = np.exp(-x**2)

# Time stepping
for n in range(nt):
    # Periodic boundary conditions
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    
    # Central difference for diffusion
    d2u_dx2 = (u_right - 2*u + u_left) / dx**2
    
    # Upwind scheme for convection
    du_dx = (u - u_left) / dx
    
    # Update solution
    u = u - dt * (c * du_dx - epsilon * d2u_dx2)

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_1D_Linear_Convection.npy', u)