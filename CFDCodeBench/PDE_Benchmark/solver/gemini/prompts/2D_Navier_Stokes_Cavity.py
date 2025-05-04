import numpy as np

def solve_lid_driven_cavity():
    # Parameters
    rho = 1.0
    nu = 0.1
    Lx = 2.0
    Ly = 2.0
    T = 10.0
    nx = 41
    ny = 41
    nt = 100
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / (nt - 1)

    # Initialize variables
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Boundary conditions
    u[ny - 1, :] = 1.0  # Top boundary (lid-driven)
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, nx - 1] = 0.0
    v[0, :] = 0.0
    v[ny - 1, :] = 0.0
    v[:, 0] = 0.0
    v[:, nx - 1] = 0.0

    # Time loop
    for n in range(nt):
        # Solve for u and v
        u_old = u.copy()
        v_old = v.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                u[i, j] = u_old[i, j] + dt * (
                    -u_old[i, j] * (u_old[i, j + 1] - u_old[i, j - 1]) / (2 * dx)
                    - v_old[i, j] * (u_old[i + 1, j] - u_old[i - 1, j]) / (2 * dy)
                    - (1 / rho) * (p[i, j + 1] - p[i, j - 1]) / (2 * dx)
                    + nu * ((u_old[i, j + 1] - 2 * u_old[i, j] + u_old[i, j - 1]) / (dx * dx) + (u_old[i + 1, j] - 2 * u_old[i, j] + u_old[i - 1, j]) / (dy * dy))
                )

                v[i, j] = v_old[i, j] + dt * (
                    -u_old[i, j] * (v_old[i, j + 1] - v_old[i, j - 1]) / (2 * dx)
                    - v_old[i, j] * (v_old[i + 1, j] - v_old[i - 1, j]) / (2 * dy)
                    - (1 / rho) * (p[i + 1, j] - p[i - 1, j]) / (2 * dy)
                    + nu * ((v_old[i, j + 1] - 2 * v_old[i, j] + v_old[i, j - 1]) / (dx * dx) + (v_old[i + 1, j] - 2 * v_old[i, j] + v_old[i - 1, j]) / (dy * dy))
                )

        # Boundary conditions for u and v
        u[ny - 1, :] = 1.0  # Top boundary (lid-driven)
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, nx - 1] = 0.0
        v[0, :] = 0.0
        v[ny - 1, :] = 0.0
        v[:, 0] = 0.0
        v[:, nx - 1] = 0.0

        # Solve for pressure
        p_old = p.copy()
        for iter in range(50):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    p[i, j] = 0.25 * (
                        p_old[i, j + 1] + p_old[i, j - 1] + p_old[i + 1, j] + p_old[i - 1, j]
                        - rho * (
                            (u[i, j + 1] - u[i, j - 1]) / (2 * dx) * (u[i, j + 1] - u[i, j - 1]) / (2 * dx)
                            + 2 * (u[i + 1, j] - u[i - 1, j]) / (2 * dy) * (v[i, j + 1] - v[i, j - 1]) / (2 * dx)
                            + (v[i + 1, j] - v[i - 1, j]) / (2 * dy) * (v[i + 1, j] - v[i - 1, j]) / (2 * dy)
                        ) * dx * dx
                    )

            # Boundary conditions for pressure
            for j in range(1, nx - 1):
                p[0, j] = p[1, j]  # dp/dy = 0 at y = 0
            for i in range(0, ny):
                p[i, 0] = p[i, 1]  # dp/dx = 0 at x = 0
                p[i, nx - 1] = p[i, nx - 2]  # dp/dx = 0 at x = 2
            p[ny - 1, :] = 0.0  # p = 0 at y = 2

    # Save the results
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Navier_Stokes_Cavity.npy', p)

solve_lid_driven_cavity()