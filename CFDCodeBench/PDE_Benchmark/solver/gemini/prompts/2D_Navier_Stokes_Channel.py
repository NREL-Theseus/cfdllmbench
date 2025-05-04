import numpy as np

# Parameters
rho = 1.0
nu = 0.1
F = 1.0
Lx = 2.0
Ly = 2.0
T = 5.0
nx = 21
ny = 21
nt = 100
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

# Initialize variables
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))
p = np.zeros((nx, ny))

# Functions for finite difference calculations
def laplacian(phi, dx, dy):
    return (
        (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2
        + (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dy**2
    )

def grad_x(phi, dx):
    return (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dx)

def grad_y(phi, dy):
    return (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dy)

# Time loop
for n in range(nt - 1):
    # Calculate intermediate velocities
    u_old = u.copy()
    v_old = v.copy()

    u[1:-1, 1:-1] = (
        u_old[1:-1, 1:-1]
        + dt
        * (
            -u_old[1:-1, 1:-1] * (u_old[2:, 1:-1] - u_old[:-2, 1:-1]) / (2 * dx)
            - v_old[1:-1, 1:-1] * (u_old[1:-1, 2:] - u_old[1:-1, :-2]) / (2 * dy)
            - (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx) / rho
            + nu * laplacian(u_old, dx, dy)
            + F
        )
    )

    v[1:-1, 1:-1] = (
        v_old[1:-1, 1:-1]
        + dt
        * (
            -u_old[1:-1, 1:-1] * (v_old[2:, 1:-1] - v_old[:-2, 1:-1]) / (2 * dx)
            - v_old[1:-1, 1:-1] * (v_old[1:-1, 2:] - v_old[1:-1, :-2]) / (2 * dy)
            - (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy) / rho
            + nu * laplacian(v_old, dx, dy)
        )
    )

    # Boundary conditions for u and v
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    u[0, :] = u[-1, :]
    u[-1, :] = u[0, :]
    v[0, :] = v[-1, :]
    v[-1, :] = v[0, :]

    # Pressure Poisson equation
    for _ in range(50):  # Iterate to convergence
        rhs = -rho * (
            grad_x(u, dx)**2
            + 2 * (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy) * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
            + grad_y(v, dy)**2
        )
        p[1:-1, 1:-1] = 0.25 * (
            p[2:, 1:-1]
            + p[:-2, 1:-1]
            + p[1:-1, 2:]
            + p[1:-1, :-2]
            - dx**2 * rhs
        )
        # Boundary conditions for pressure
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[-1, :]
        p[-1, :] = p[0, :]

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Navier_Stokes_Channel.npy', p)