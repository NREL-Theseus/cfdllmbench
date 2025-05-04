#!/usr/bin/env python3
import numpy as np

# Domain and simulation parameters
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0
t_start, t_end = 0.0, 0.5

# Number of grid points
nx = 101
ny = 101
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)

# Convection speed
c = 1.0

# CFL number and time step size (upwind scheme: dt <= min(dx,dy)/c)
CFL = 0.4
dt = CFL * min(dx, dy) / c

# Create grid
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition: u = 2 for 0.5 <= x <= 1 and 0.5 <= y <= 1, otherwise 1.
u = np.ones((nx, ny))
u[np.logical_and(X >= 0.5, X <= 1.0) & np.logical_and(Y >= 0.5, Y <= 1.0)] = 2.0

# Enforce Dirichlet boundary conditions (u=1 on boundaries)
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0

# Time integration using explicit upwind finite differences
t = t_start
while t < t_end:
    # Ensure we do not overshoot final time
    if t + dt > t_end:
        dt = t_end - t
        
    u_new = u.copy()
    
    # Update interior points only (from i=1 to nx-1, j=1 to ny-1)
    # Use first order upwind scheme for positive c: backward difference in x and y.
    u_new[1:, 1:] = ( u[1:, 1:] 
                      - c * dt / dx * (u[1:, 1:] - u[:-1, 1:])
                      - c * dt / dy * (u[1:, 1:] - u[1:, :-1]) )
    
    # Enforce Dirichlet boundary conditions on all boundaries
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0

    # Update solution and time
    u = u_new.copy()
    t += dt

# Save the final solution u (a 2D array) as u.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_2D_Linear_Convection.npy', u)