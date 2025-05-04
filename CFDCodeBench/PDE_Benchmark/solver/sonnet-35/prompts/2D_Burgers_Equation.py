import numpy as np
import matplotlib.pyplot as plt

# Problem Parameters
Lx, Ly = 2.0, 2.0  # Domain size
nu = 0.01  # Kinematic viscosity
T = 0.027  # Total simulation time

# Discretization
nx, ny = 100, 100  # Number of grid points
dx, dy = Lx / (nx-1), Ly / (ny-1)
nt, dt = 1000, T / 1000  # Time steps

# Initialize solution arrays
u = np.ones((ny, nx)) 
v = np.ones((ny, nx))

# Initial condition modification
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

mask = ((X >= 0.5) & (X <= 1.0) & 
        (Y >= 0.5) & (Y <= 1.0))
u[mask] = 2.0
v[mask] = 2.0

# Boundary conditions
u[0,:], u[-1,:] = 1.0, 1.0
u[:,0], u[:,-1] = 1.0, 1.0
v[0,:], v[-1,:] = 1.0, 1.0
v[:,0], v[:,-1] = 1.0, 1.0

# Time integration (Finite Difference Method)
for _ in range(nt):
    # Create copies to store previous time step
    un = u.copy()
    vn = v.copy()
    
    # Interior points update
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            # u-momentum equation
            u[i,j] = (un[i,j] - 
                      un[i,j] * dt/dx * (un[i,j] - un[i,j-1]) - 
                      vn[i,j] * dt/dy * (un[i,j] - un[i-1,j]) + 
                      nu * dt/dx**2 * (un[i,j+1] - 2*un[i,j] + un[i,j-1]) + 
                      nu * dt/dy**2 * (un[i+1,j] - 2*un[i,j] + un[i-1,j]))
            
            # v-momentum equation
            v[i,j] = (vn[i,j] - 
                      un[i,j] * dt/dx * (vn[i,j] - vn[i,j-1]) - 
                      vn[i,j] * dt/dy * (vn[i,j] - vn[i-1,j]) + 
                      nu * dt/dx**2 * (vn[i,j+1] - 2*vn[i,j] + vn[i,j-1]) + 
                      nu * dt/dy**2 * (vn[i+1,j] - 2*vn[i,j] + vn[i-1,j]))
    
    # Enforce boundary conditions
    u[0,:], u[-1,:] = 1.0, 1.0
    u[:,0], u[:,-1] = 1.0, 1.0
    v[0,:], v[-1,:] = 1.0, 1.0
    v[:,0], v[:,-1] = 1.0, 1.0

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Burgers_Equation.npy', v)