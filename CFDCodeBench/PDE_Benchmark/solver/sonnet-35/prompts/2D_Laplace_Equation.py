import numpy as np

# Problem parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50  # Grid resolution
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize potential field
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = np.linspace(0, 1, ny)  # Right boundary with y-dependent condition

# Solve using finite difference method (Jacobi iteration)
max_iter = 10000
tolerance = 1e-6

for _ in range(max_iter):
    p_old = p.copy()
    
    # Update interior points using finite difference approximation
    p[1:-1, 1:-1] = 0.25 * (
        p_old[1:-1, 2:] + 
        p_old[1:-1, :-2] + 
        p_old[2:, 1:-1] + 
        p_old[:-2, 1:-1]
    )
    
    # Enforce Neumann conditions at top and bottom
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    
    # Check convergence
    if np.max(np.abs(p - p_old)) < tolerance:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Laplace_Equation.npy', p)