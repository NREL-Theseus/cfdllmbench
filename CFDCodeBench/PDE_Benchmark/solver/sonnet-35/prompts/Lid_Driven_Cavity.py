import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain dimensions
rho = 1.0  # Density
nu = 0.1  # Kinematic viscosity
dt = 0.001  # Time step
T = 0.5  # Total simulation time
nx, ny = 50, 50  # Grid resolution
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Grid generation
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Boundary conditions
u[-1, :] = 1.0  # Top lid moving with unit velocity

# Time-stepping using Fractional Step Method
def compute_intermediate_velocity(u, v):
    u_star = u.copy()
    v_star = v.copy()
    
    # Advection terms (simple central difference)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u_star[i, j] = u[i, j] - dt * (
                u[i, j] * (u[i, j+1] - u[i, j-1]) / (2*dx) + 
                v[i, j] * (u[i+1, j] - u[i-1, j]) / (2*dy)
            ) + nu * dt * (
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2 +
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2
            )
            
            v_star[i, j] = v[i, j] - dt * (
                u[i, j] * (v[i, j+1] - v[i, j-1]) / (2*dx) + 
                v[i, j] * (v[i+1, j] - v[i-1, j]) / (2*dy)
            ) + nu * dt * (
                (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / dx**2 +
                (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / dy**2
            )
    
    # Enforce boundary conditions
    u_star[0, :] = 0
    u_star[-1, :] = 1.0
    u_star[:, 0] = 0
    u_star[:, -1] = 0
    
    v_star[0, :] = 0
    v_star[-1, :] = 0
    v_star[:, 0] = 0
    v_star[:, -1] = 0
    
    return u_star, v_star

def solve_pressure_poisson(u_star, v_star):
    # Compute divergence of intermediate velocity
    div = np.zeros_like(p)
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            div[i, j] = (u_star[i, j+1] - u_star[i, j-1]) / (2*dx) + \
                        (v_star[i+1, j] - v_star[i-1, j]) / (2*dy)
    
    # Solve Poisson equation for pressure using Jacobi iteration
    p_new = p.copy()
    for _ in range(100):  # Fixed number of iterations
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                p_new[i, j] = 0.25 * (
                    p[i, j+1] + p[i, j-1] + 
                    p[i+1, j] + p[i-1, j] - 
                    dx**2 * div[i, j] / dt
                )
        
        # Neumann boundary conditions for pressure
        p_new[0, :] = p_new[1, :]
        p_new[-1, :] = p_new[-2, :]
        p_new[:, 0] = p_new[:, 1]
        p_new[:, -1] = p_new[:, -2]
        
        p = p_new.copy()
    
    return p

def correct_velocity(u_star, v_star, p):
    u = u_star.copy()
    v = v_star.copy()
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            u[i, j] = u_star[i, j] - dt * (p[i, j+1] - p[i, j-1]) / (2*dx*rho)
            v[i, j] = v_star[i, j] - dt * (p[i+1, j] - p[i-1, j]) / (2*dy*rho)
    
    # Enforce boundary conditions
    u[0, :] = 0
    u[-1, :] = 1.0
    u[:, 0] = 0
    u[:, -1] = 0
    
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    
    return u, v

# Time-stepping loop
t = 0
while t < T:
    # Compute intermediate velocity
    u_star, v_star = compute_intermediate_velocity(u, v)
    
    # Solve pressure Poisson equation
    p = solve_pressure_poisson(u_star, v_star)
    
    # Correct velocities
    u, v = correct_velocity(u_star, v_star, p)
    
    t += dt

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_Lid_Driven_Cavity.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/v_Lid_Driven_Cavity.npy', v)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_Lid_Driven_Cavity.npy', p)