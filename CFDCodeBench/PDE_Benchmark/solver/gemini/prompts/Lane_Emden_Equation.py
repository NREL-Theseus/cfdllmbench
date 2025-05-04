import numpy as np

def solve_lane_emden(n=3.0, r_max=1.0, num_points=100):
    """
    Solves the Lane-Emden equation using a finite difference method.

    Args:
        n (float): Polytropic index.
        r_max (float): Outer radius.
        num_points (int): Number of radial points.

    Returns:
        numpy.ndarray: Solution f(r) at the final iteration.
    """

    r = np.linspace(0, r_max, num_points)
    dr = r[1] - r[0]

    # Initial guess
    R0 = 5
    f = R0**(2/(n-1)) * (1 - r**2)**2

    # Boundary condition
    f[-1] = 0.0

    # Iterative solver (Newton-Raphson)
    tolerance = 1e-6
    max_iterations = 1000
    error = 1.0
    iteration = 0

    while error > tolerance and iteration < max_iterations:
        f_old = f.copy()

        # Finite difference discretization
        A = np.zeros((num_points, num_points))
        b = np.zeros(num_points)

        # Interior points
        for i in range(1, num_points - 1):
            A[i, i-1] = 1 - dr / (2 * r[i])
            A[i, i] = -2 + dr**2 * f[i]**n
            A[i, i+1] = 1 + dr / (2 * r[i])
            b[i] = -dr**2 * f[i]**(n+1)

        # Boundary condition at r = r_max
        A[-1, -1] = 1
        b[-1] = 0

        # Regularity condition at r = 0 (L'Hopital's rule)
        A[0, 0] = -2
        A[0, 1] = 2
        b[0] = 0

        # Solve the linear system
        delta_f = np.linalg.solve(A, b)
        f = f - delta_f

        # Update error and iteration
        error = np.max(np.abs(f - f_old))
        iteration += 1

    return f

if __name__ == "__main__":
    # Solve the Lane-Emden equation
    f_solution = solve_lane_emden()

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/f_solution_Lane_Emden_Equation.npy', f_solution)