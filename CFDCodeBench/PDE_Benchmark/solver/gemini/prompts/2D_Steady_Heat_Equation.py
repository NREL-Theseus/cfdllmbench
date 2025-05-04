import numpy as np

def solve_heat_equation():
    # Domain parameters
    width = 5.0
    height = 4.0

    # Grid parameters
    nx = 50
    ny = 40
    dx = width / (nx - 1)
    dy = height / (ny - 1)

    # Boundary conditions
    T_left = 10.0
    T_right = 40.0
    T_top = 0.0
    T_bottom = 20.0

    # Initial condition
    T = np.zeros((ny, nx))

    # Apply boundary conditions
    T[:, 0] = T_left
    T[:, -1] = T_right
    T[0, :] = T_bottom
    T[-1, :] = T_top

    # Iterative solution (Jacobi method)
    max_iterations = 10000
    tolerance = 1.0e-6
    error = 1.0
    iteration = 0

    while error > tolerance and iteration < max_iterations:
        T_old = T.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                T[i, j] = 0.25 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1])
        error = np.max(np.abs(T - T_old))
        iteration += 1

    # Save the temperature field
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/T_2D_Steady_Heat_Equation.npy', T)

solve_heat_equation()