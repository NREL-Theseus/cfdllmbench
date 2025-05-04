import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
Lx, Ly = 2.0, 2.0
rho = 1.0
nu = 0.1
F = 1.0
T = 5.0

# Discretization
nx, ny = 50, 50
nt = 200
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / nt

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Time-stepping (Finite Difference Method)
for n in range(nt):
    # Store old values
    u_old = u.copy()
    v_old = v.copy()
    
    # Compute derivatives
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Diffusion terms
    d2u_dx2 = np.gradient(np.gradient(u, dx, axis=1), dx, axis=1)
    d2u_dy2 = np.gradient(np.gradient(u, dy, axis=0), dy, axis=0)
    d2v_dx2 = np.gradient(np.gradient(v, dx, axis=1), dx, axis=1)
    d2v_dy2 = np.gradient(np.gradient(v, dy, axis=0), dy, axis=0)
    
    # Pressure Poisson equation
    div_u = du_dx + dv_dy
    p_rhs = -rho * (du_dx**2 + 2*du_dy*dv_dx + dv_dy**2)
    
    # Solve pressure Poisson equation
    p_laplacian = sp.poisson((ny, nx), format='csr')
    p = spla.spsolve(p_laplacian, p_rhs.ravel()).reshape((ny, nx))
    
    # Pressure gradient
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)
    
    # Update velocity fields
    u[1:-1, 1:-1] = (u_old[1:-1, 1:-1] 
                     - dt * (u_old[1:-1, 1:-1] * du_dx[1:-1, 1:-1] 
                             + v_old[1:-1, 1:-1] * du_dy[1:-1, 1:-1])
                     + dt * nu * (d2u_dx2[1:-1, 1:-1] + d2u_dy2[1:-1, 1:-1])
                     - dt/rho * dp_dx[1:-1, 1:-1]
                     + dt * F)
    
    v[1:-1, 1:-1] = (v_old[1:-1, 1:-1] 
                     - dt * (u_old[1:-1, 1:-1] * dv_dx[1:-1, 1:-1] 
                             + v_old[1:-1, 1:-1] * dv_dy[1:-1, 1:-1])
                     + dt * nu * (d2v_dx2[1:-1, 1:-1] + d2v_dy2[1:-1, 1:-1])
                     - dt/rho * dp_dy[1:-1, 1:-1])
    
    # Periodic boundary conditions in x
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    p[:, 0] = p[:, -2]
    p[:, -1] = p[:, 1]
    
    # No-slip boundary conditions in y
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :] = 0

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Navier_Stokes_Channel.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Navier_Stokes_Channel.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Navier_Stokes_Channel.npy', p)