import numpy as np

def solve_pde():
    # Parameters
    nx = 50
    ny = 50
    nt = 50
    c = 1.0
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = 0.01  # Choose dt to satisfy CFL condition (dt <= dx/c)

    # Initialize u
    u = np.ones((ny, nx))
    for i in range(ny):
        for j in range(nx):
            x = j * dx
            y = i * dy
            if 0.5 <= x <= 1.0 and 0.5 <= y <= 1.0:
                u[i, j] = 2.0

    # Boundary conditions
    u[:, 0] = 1.0  # x = 0
    u[:, -1] = 1.0 # x = 2
    u[0, :] = 1.0  # y = 0
    u[-1, :] = 1.0 # y = 2

    # Numerical solution using upwind scheme
    for n in range(nt):
        u_old = u.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = u_old[i, j] - c * dt / dx * (u_old[i, j] - u_old[i, j - 1]) - c * dt / dy * (u_old[i, j] - u_old[i - 1, j])

        # Boundary conditions (re-apply at each time step)
        u[:, 0] = 1.0  # x = 0
        u[:, -1] = 1.0 # x = 2
        u[0, :] = 1.0  # y = 0
        u[-1, :] = 1.0 # y = 2

    # Save the solution at the final time step
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Linear_Convection.npy', u)

solve_pde()