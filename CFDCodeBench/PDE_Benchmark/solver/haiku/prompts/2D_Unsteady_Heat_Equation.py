import numpy as np

# Parameters
alpha = 0.1  # thermal diffusivity
Q0 = 200     # source term amplitude 
sigma = 0.1  # source term width

# Grid parameters
nx = ny = 101
dx = dy = 2.0/(nx-1)
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
t_final = 3.0
dt = 0.0001  # time step
nt = int(t_final/dt)

# Initialize temperature field
T = 1 + 200*np.exp(-(X**2 + Y**2)/(2*sigma**2))

# Source term
def q(X, Y):
    return Q0*np.exp(-(X**2 + Y**2)/(2*sigma**2))

# Time stepping
for n in range(nt):
    T_old = T.copy()
    
    # Interior points
    T[1:-1,1:-1] = T_old[1:-1,1:-1] + dt*(
        alpha*((T_old[1:-1,2:] - 2*T_old[1:-1,1:-1] + T_old[1:-1,:-2])/dx**2 +
              (T_old[2:,1:-1] - 2*T_old[1:-1,1:-1] + T_old[:-2,1:-1])/dy**2) +
        q(X,Y)[1:-1,1:-1])
    
    # Boundary conditions
    T[0,:] = 1  # bottom
    T[-1,:] = 1 # top
    T[:,0] = 1  # left
    T[:,-1] = 1 # right

# Save final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/T_2D_Unsteady_Heat_Equation.npy', T)