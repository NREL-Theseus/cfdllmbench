import numpy as np

def solve_pde():
    # Parameters
    nx = 41
    ny = 41
    nt = 100
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 2
    tmax = 0.32
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dt = tmax / (nt - 1)

    # Initialize arrays
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))

    # Initial conditions
    for i in range(ny):
        for j in range(nx):
            x = xmin + j * dx
            y = ymin + i * dy
            if 0.5 <= x <= 1 and 0.5 <= y <= 1:
                u[i, j] = 2
                v[i, j] = 2

    # Boundary conditions (Dirichlet)
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1

    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

    # Time loop
    for n in range(nt - 1):
        u_old = u.copy()
        v_old = v.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = u_old[i, j] - dt * (u_old[i, j] * (u_old[i, j] - u_old[i, j - 1]) / dx +
                                                v_old[i, j] * (u_old[i, j] - u_old[i - 1, j]) / dy)
                v[i, j] = v_old[i, j] - dt * (u_old[i, j] * (v_old[i, j] - v_old[i, j - 1]) / dx +
                                                v_old[i, j] * (v_old[i, j] - v_old[i - 1, j]) / dy)

        # Boundary conditions (Dirichlet)
        u[:, 0] = 1
        u[:, -1] = 1
        u[0, :] = 1
        u[-1, :] = 1

        v[:, 0] = 1
        v[:, -1] = 1
        v[0, :] = 1
        v[-1, :] = 1

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Convection.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Convection.npy', v)

solve_pde()