#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 41
ny = 41
nt = 500        # number of time steps such that dt * nt = 0.5 (dt = 0.001)
nit = 50        # number of iterations for pressure Poisson equation per time step
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
dt = 0.001

rho = 1.0
nu = 0.1

# Initialize fields: 2D arrays for u, v, pressure p
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))  # source term for pressure Poisson

# Function to build the source term for pressure Poisson
def build_up_b(b, u, v, dx, dy, dt, rho):
    b[1:-1, 1:-1] = (rho * (1.0/dt *
                     ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) +
                      (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)) -
                     ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 -
                       2 * ((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy) *
                            (v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) -
                         ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2))
    return b

# Function to solve the pressure Poisson equation
def pressure_poisson(p, dx, dy, b, nit):
    for _ in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                        (2*(dx**2+dy**2)) -
                        dx**2*dy**2/(2*(dx**2+dy**2)) * b[1:-1,1:-1])
        # Boundary conditions: dp/dn=0 at walls
        p[:, -1] = p[:, -2]   # right wall
        p[:, 0]  = p[:, 1]    # left wall
        p[-1, :] = p[-2, :]   # top wall
        p[0, :]  = p[1, :]    # bottom wall
        # Optionally, set a reference pressure point to 0 for uniqueness
        p[0,0] = 0.0
    return p

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Build up pressure RHS source term
    b = build_up_b(b, un, vn, dx, dy, dt, rho)
    
    # Compute intermediate velocity u_star and v_star (explicit update)
    # Using central differences for convection and diffusion terms
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt/dx * (un[1:-1,1:-1] - un[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt/dy * (un[1:-1,1:-1] - un[0:-2,1:-1]) +
                    nu * dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) +
                    nu * dt/dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    un[1:-1,1:-1] * dt/dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) -
                    vn[1:-1,1:-1] * dt/dy * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) +
                    nu * dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) +
                    nu * dt/dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]))
    
    # Apply boundary conditions for velocity: no-slip on walls, lid-driven on top wall
    # No-slip on bottom and side walls
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    v[-1, :] = 0

    # Top lid: u=1, v=0 (override computed u on top border)
    u[-1, :] = 1
    v[-1, :] = 0

    # Solve Pressure Poisson equation to enforce incompressibility
    p = pressure_poisson(p, dx, dy, b, nit)
    
    # Pressure correction for velocity (projection step)
    u[1:-1,1:-1] -= dt/(2*rho*dx) * (p[1:-1,2:] - p[1:-1,0:-2])
    v[1:-1,1:-1] -= dt/(2*rho*dy) * (p[2:,1:-1] - p[0:-2,1:-1])
    
    # Reapply boundary conditions for velocity after correction
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1  # top lid
    v[0, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    v[-1, :] = 0

# Save final solution fields as .npy files (2D arrays)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_Lid_Driven_Cavity.npy', p)