import numpy as np

# Domain parameters
Lx = 2.0
Ly = 1.0
nx = 100  # number of points in x
ny = 50   # number of points in y

# Grid setup
dx = Lx/(nx-1)
dy = Ly/(ny-1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize pressure field
p = np.zeros((ny, nx))

# Source term
b = np.zeros((ny, nx))
source_x1, source_y1 = int(0.25*nx), int(0.25*ny)
source_x2, source_y2 = int(0.75*nx), int(0.75*ny)
b[source_y1, source_x1] = 100
b[source_y2, source_x2] = -100

# Iteration parameters
max_iter = 10000
tolerance = 1e-6

# Gauss-Seidel iteration
for it in range(max_iter):
    p_old = p.copy()
    
    # Update interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            p[i,j] = ((p[i+1,j] + p[i-1,j])/dy**2 + 
                      (p[i,j+1] + p[i,j-1])/dx**2 - 
                      b[i,j]) / (2/dx**2 + 2/dy**2)
    
    # Apply boundary conditions
    p[0,:] = 0  # bottom
    p[-1,:] = 0 # top
    p[:,0] = 0  # left
    p[:,-1] = 0 # right
    
    # Check convergence
    error = np.max(np.abs(p - p_old))
    if error < tolerance:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Poisson_Equation.npy', p)