import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx, Ly = 2.0, 2.0  # Domain size
nx, ny = 100, 100  # Grid points
nt = 100  # Time steps
dt = 0.32 / nt  # Time step size
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize solution arrays
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition modification
mask = (0.5 <= np.linspace(0, Lx, nx)) & (np.linspace(0, Ly, ny)[:, np.newaxis] >= 0.5)
u[mask] = 2
v[mask] = 2

# Finite difference method (MacCormack scheme)
def macCormack2D(u, v):
    u_pred = np.copy(u)
    v_pred = np.copy(v)
    
    # Predictor step
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_pred[j,i] = u[j,i] - dt * (u[j,i] * (u[j,i+1] - u[j,i-1])/(2*dx) + 
                                          v[j,i] * (u[j+1,i] - u[j-1,i])/(2*dy))
            v_pred[j,i] = v[j,i] - dt * (u[j,i] * (v[j,i+1] - v[j,i-1])/(2*dx) + 
                                          v[j,i] * (v[j+1,i] - v[j-1,i])/(2*dy))
    
    # Corrector step
    u_corr = np.copy(u)
    v_corr = np.copy(v)
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_corr[j,i] = 0.5 * (u[j,i] + u_pred[j,i] - 
                                  dt * (u_pred[j,i] * (u_pred[j,i+1] - u_pred[j,i-1])/(2*dx) + 
                                        v_pred[j,i] * (u_pred[j+1,i] - u_pred[j-1,i])/(2*dy)))
            v_corr[j,i] = 0.5 * (v[j,i] + v_pred[j,i] - 
                                  dt * (u_pred[j,i] * (v_pred[j,i+1] - v_pred[j,i-1])/(2*dx) + 
                                        v_pred[j,i] * (v_pred[j+1,i] - v_pred[j-1,i])/(2*dy)))
    
    return u_corr, v_corr

# Time integration
for _ in range(nt):
    # Apply boundary conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1
    
    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1
    
    # Update solution
    u, v = macCormack2D(u, v)

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_2D_Convection.npy', v)