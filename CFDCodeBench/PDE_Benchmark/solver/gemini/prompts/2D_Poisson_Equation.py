import numpy as np

def solve_poisson(Lx, Ly, nx, ny):
    """
    Solves the Poisson equation using the finite difference method.

    Args:
        Lx (float): Length of the domain in the x-direction.
        Ly (float): Length of the domain in the y-direction.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.

    Returns:
        ndarray: Solution of the Poisson equation.
    """

    # Grid spacing
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Initialize the solution and source term
    p = np.zeros((nx, ny))
    b = np.zeros((nx, ny))

    # Define the source term
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    b[(abs(X - Lx/4) < dx/2) & (abs(Y - Ly/4) < dy/2)] = 100
    b[(abs(X - 3*Lx/4) < dx/2) & (abs(Y - 3*Ly/4) < dy/2)] = -100

    # Apply Dirichlet boundary conditions
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0

    # Iterate until convergence
    max_iter = 10000
    tolerance = 1e-6
    error = 1.0
    iter_count = 0

    while error > tolerance and iter_count < max_iter:
        p_old = np.copy(p)

        # Update the interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                p[i, j] = 0.25 * (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - b[i, j] * dx**2)

        # Apply Dirichlet boundary conditions
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0

        # Calculate the error
        error = np.max(np.abs(p - p_old))
        iter_count += 1

    return p

if __name__ == '__main__':
    # Define the problem parameters
    Lx = 2.0
    Ly = 1.0
    nx = 50
    ny = 50

    # Solve the Poisson equation
    p = solve_poisson(Lx, Ly, nx, ny)

    # Save the solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Poisson_Equation.npy', p)