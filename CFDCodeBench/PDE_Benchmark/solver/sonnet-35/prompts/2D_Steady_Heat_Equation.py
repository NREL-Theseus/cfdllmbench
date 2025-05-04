import numpy as np

# Domain parameters
Lx, Ly = 5.0, 4.0
nx, ny = 100, 80  # Grid resolution

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx, dy = x[1] - x[0], y[1] - y[0]

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary

# Solve using finite difference method (Jacobi iteration)
max_iter = 10000
tolerance = 1e-6

for _ in range(max_iter):
    T_old = T.copy()
    
    # Update interior points using finite difference approximation
    T[1:-1, 1:-1] = 0.25 * (
        T_old[1:-1, 2:] + 
        T_old[1:-1, :-2] + 
        T_old[2:, 1:-1] + 
        T_old[:-2, 1:-1]
    )
    
    # Check convergence
    if np.max(np.abs(T - T_old)) < tolerance:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/T_2D_Steady_Heat_Equation.npy', T)