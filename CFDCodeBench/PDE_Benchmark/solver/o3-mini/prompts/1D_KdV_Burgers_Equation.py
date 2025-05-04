#!/usr/bin/env python
import numpy as np

# Parameters
L = 10.0              # domain length
T = 10.0              # final time
a = 1e-4              # diffusion coefficient
b = 2e-4              # dispersion coefficient
n_param = 20          # parameter for initial condition

# Numerical parameters
Nx = 256                         # number of spatial grid points
dx = L / Nx                      # spatial step size
dt = 1e-3                        # time step size
nt = int(T / dt)                 # number of time steps

# Spatial grid
x = np.linspace(0, L, Nx, endpoint=False)

# Initial condition
u = (1/(2*n_param)) * np.log(1 + (np.cosh(n_param)**2) / (np.cosh(n_param*(x - 0.2*L))**2))

# Periodic finite difference operators using np.roll
def compute_rhs(u):
    # First derivative: central difference
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)
    # Second derivative: central difference
    uxx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
    # Third derivative: central difference (5-point stencil)
    uxxx = (-np.roll(u, 2) + 2*np.roll(u, 1) - 2*np.roll(u, -1) + np.roll(u, -2)) / (2*dx**3)
    # KdV-Burgers equation: u_t + u*u_x = a*u_xx + b*u_xxx
    # Rearranged: u_t = - u*u_x + a*u_xx + b*u_xxx
    return - u * ux + a * uxx + b * uxxx

# Time integration using 4th order Runge-Kutta
for _ in range(nt):
    k1 = compute_rhs(u)
    k2 = compute_rhs(u + 0.5*dt*k1)
    k3 = compute_rhs(u + 0.5*dt*k2)
    k4 = compute_rhs(u + dt*k3)
    u = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Save the final solution in a 1D NumPy array
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_1D_KdV_Burgers_Equation.npy', u)