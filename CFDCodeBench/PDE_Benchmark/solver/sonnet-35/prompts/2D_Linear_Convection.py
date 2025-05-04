import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx, Ly = 2.0, 2.0  # Domain dimensions
T = 0.50  # Total simulation time
c = 1.0  # Convection speed
nx, ny = 100, 100  # Number of grid points
nt = 500  # Number of time steps

# Grid generation
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize solution array
u = np.ones((ny, nx))

# Initial condition
mask = ((x >= 0.5) & (x <= 1.0))
mask2d_x, mask2d_y = np.meshgrid(mask, mask)
u[(0.5 <= x[:, np.newaxis]) & (x[:, np.newaxis] <= 1.0) & 
  (0.5 <= y[:, np.newaxis]) & (y[:, np.newaxis] <= 1.0)] = 2.0

# Finite difference method (upwind scheme)
for _ in range(nt):
    u_old = u.copy()
    
    # Interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u[i,j] = u_old[i,j] - c * dt/dx * (u_old[i,j] - u_old[i,j-1]) \
                               - c * dt/dy * (u_old[i,j] - u_old[i-1,j])
    
    # Boundary conditions
    u[0,:] = 1.0  # Bottom boundary
    u[-1,:] = 1.0  # Top boundary
    u[:,0] = 1.0  # Left boundary
    u[:,-1] = 1.0  # Right boundary

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Linear_Convection.npy', u)