#!/usr/bin/env python3
import numpy as np

# Parameters
c = 1.0                   # convection speed
epsilon = 5e-4            # diffusion/damping factor (set to 0 for undamped, 5e-4 for damped)
x_start = -5.0
x_end = 5.0
t_final = 10.0

# Discretization
Nx = 501                      # number of spatial points
x = np.linspace(x_start, x_end, Nx)
dx = x[1] - x[0]
dt = 0.005                   # time step, chosen to satisfy CFL conditions
n_steps = int(t_final/dt)

# Initial condition: u(x,0) = exp(-x^2)
u = np.exp(-x**2)

# Time integration: explicit finite difference method with upwind for convection and central differences for diffusion
for step in range(n_steps):
    # Periodic boundary: use np.roll for neighbors
    u_left = np.roll(u, 1)   # u_{i-1}
    u_right = np.roll(u, -1) # u_{i+1}
    
    # Upwind scheme for convection term (since c>0, use u - u_left)
    convective = (u - u_left) / dx
    
    # Central difference for diffusion term
    diffusive = (u_right - 2*u + u_left) / (dx**2)
    
    # Update equation
    u = u - c * dt * convective + epsilon * dt * diffusive

# Save final solution as a 1D numpy array in 'u.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_Linear_Convection.npy', u)