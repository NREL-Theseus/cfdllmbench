import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
Lx, Ly = 2.0, 2.0
rho = 1.0
nu = 0.1
t_end = 10.0

# Discretization parameters
nx, ny = 50, 50
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = 0.01

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time-stepping
def compute_derivatives(field):
    dx_field = np.zeros_like(field)
    dy_field = np.zeros_like(field)
    
    # Central difference for interior points
    dx_field[1:-1, 1:-1] = (field[1:-1, 2:] - field[1:-1, :-2]) / (2*dx)
    dy_field[1:-1, 1:-1] = (field[2:, 1:-1] - field[:-2, 1:-1]) / (2*dy)
    
    return dx_field, dy_field

def laplacian(field):
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = ((field[1:-1, 2:] + field[1:-1, :-2] - 2*field[1:-1, 1:-1]) / dx**2 +
                       (field[2:, 1:-1] + field[:-2, 1:-1] - 2*field[1:-1, 1:-1]) / dy**2)
    return lap

# Time integration
for t in np.arange(0, t_end, dt):
    # Apply boundary conditions
    u[0, :] = 0  # Bottom wall
    u[-1, :] = 1  # Top wall (lid-driven)
    u[:, 0] = 0  # Left wall
    u[:, -1] = 0  # Right wall
    
    v[0, :] = 0  # Bottom wall
    v[-1, :] = 0  # Top wall
    v[:, 0] = 0  # Left wall
    v[:, -1] = 0  # Right wall
    
    # Compute derivatives
    du_dx, du_dy = compute_derivatives(u)
    dv_dx, dv_dy = compute_derivatives(v)
    
    # Momentum equations
    u_new = u + dt * (-u * du_dx - v * du_dy + 
                      nu * laplacian(u))
    
    v_new = v + dt * (-u * dv_dx - v * dv_dy + 
                      nu * laplacian(v))
    
    # Pressure Poisson equation
    rhs = -rho * (du_dx**2 + 2*du_dy*dv_dx + dv_dy**2)
    
    # Solve pressure Poisson equation
    p_new = np.zeros_like(p)
    for _ in range(100):  # Iterative solution
        p_old = p_new.copy()
        p_new[1:-1, 1:-1] = 0.25 * (
            p_new[1:-1, 2:] + p_new[1:-1, :-2] + 
            p_new[2:, 1:-1] + p_new[:-2, 1:-1] - 
            dx**2 * rhs[1:-1, 1:-1]
        )
        
        # Pressure boundary conditions
        p_new[0, :] = p_new[1, :]  # Neumann at bottom
        p_new[-1, :] = 0  # Dirichlet at top
        p_new[:, 0] = p_new[:, 1]  # Neumann at left
        p_new[:, -1] = p_new[:, -2]  # Neumann at right
        
        if np.max(np.abs(p_new - p_old)) < 1e-6:
            break
    
    # Update fields
    u = u_new
    v = v_new
    p = p_new

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Navier_Stokes_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Navier_Stokes_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Navier_Stokes_Cavity.npy', p)