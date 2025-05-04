import numpy as np

# Problem parameters
nu = 0.3  # Diffusion coefficient
L = 2.0   # Domain length
T = 0.0333  # Total simulation time

# Discretization parameters
nx = 100  # Number of spatial points 
nt = 200  # Number of time steps

# Grid generation
dx = L / (nx - 1)
dt = T / (nt - 1)

x = np.linspace(0, L, nx)

# Initial condition
u = np.ones(nx)
u[(x >= 0.5) & (x <= 1.0)] = 2.0

# Stability check (CFL condition)
stability_condition = nu * dt / (dx**2)
print(f"Stability condition: {stability_condition}")

# Time-stepping (Explicit Finite Difference)
for n in range(nt):
    u_old = u.copy()
    
    # Interior points
    u[1:-1] = u_old[1:-1] + nu * dt / (dx**2) * \
              (u_old[2:] - 2*u_old[1:-1] + u_old[:-2])
    
    # Neumann boundary conditions (zero-gradient)
    u[0] = u[1]
    u[-1] = u[-2]

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_1D_Diffusion.npy', u)