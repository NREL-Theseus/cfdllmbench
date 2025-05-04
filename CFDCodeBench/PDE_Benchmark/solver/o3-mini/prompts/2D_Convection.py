#!/usr/bin/env python3
import numpy as np

# Domain parameters
nx = 81
ny = 81
xmin, xmax = 0.0, 2.0
ymin, ymax = 0.0, 2.0
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

# Temporal parameters
t_final = 0.32
dt = 0.001  # time step chosen to satisfy CFL condition approximately
nt = int(t_final / dt)

# Create spatial grid
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initialize u and v fields with initial conditions
u = np.ones((nx, ny))
v = np.ones((nx, ny))
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2.0
v[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2.0

# Enforce Dirichlet boundaries for initial condition (all boundaries = 1)
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0
v[0, :] = 1.0
v[-1, :] = 1.0
v[:, 0] = 1.0
v[:, -1] = 1.0

# Time-stepping loop
for n in range(nt):
    u_old = u.copy()
    v_old = v.copy()
    
    # Update interior points using upwind finite difference (assuming u and v > 0)
    # Use indices 1 to -2 to avoid boundaries.
    u[1:-1,1:-1] = u_old[1:-1,1:-1] - dt * (
        u_old[1:-1,1:-1] * (u_old[1:-1,1:-1] - u_old[0:-2,1:-1]) / dx +
        v_old[1:-1,1:-1] * (u_old[1:-1,1:-1] - u_old[1:-1,0:-2]) / dy
    )
    
    v[1:-1,1:-1] = v_old[1:-1,1:-1] - dt * (
        u_old[1:-1,1:-1] * (v_old[1:-1,1:-1] - v_old[0:-2,1:-1]) / dx +
        v_old[1:-1,1:-1] * (v_old[1:-1,1:-1] - v_old[1:-1,0:-2]) / dy
    )
    
    # Enforce Dirichlet boundary conditions on all boundaries (u = 1, v = 1)
    u[0, :] = 1.0
    u[-1, :] = 1.0
    u[:, 0] = 1.0
    u[:, -1] = 1.0
    v[0, :] = 1.0
    v[-1, :] = 1.0
    v[:, 0] = 1.0
    v[:, -1] = 1.0

# Save final time step solutions as .npy files
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_2D_Convection.npy', v)