import numpy as np
from scipy.sparse import diags, linalg
from scipy.sparse.linalg import spsolve

# Parameters
Lx, Lz = 4.0, 1.0
Ra = 2e6
Pr = 1.0
nu = (Ra/Pr)**(-0.5)  
kappa = (Ra*Pr)**(-0.5)
T = 50.0

# Grid
nx, nz = 128, 64
dx = Lx/nx
dz = Lz/nz
x = np.linspace(0, Lx, nx)
z = np.linspace(0, Lz, nz)
X, Z = np.meshgrid(x, z, indexing='ij')

# Time stepping
dt = 0.001
nt = int(T/dt)

# Initialize fields
u = np.zeros((nx, nz))
w = np.zeros((nx, nz))
p = np.zeros((nx, nz))
b = Lz - Z + 0.01*np.random.randn(nx, nz)

# Operators for pressure Poisson equation
dx2 = dx**2
dz2 = dz**2

def pressure_solve(u, w):
    # RHS of pressure Poisson equation
    rhs = np.zeros((nx, nz))
    rhs[1:-1,1:-1] = (
        (u[2:,1:-1] - u[:-2,1:-1])/(2*dx) + 
        (w[1:-1,2:] - w[1:-1,:-2])/(2*dz)
    )/dt
    
    # Solve using successive over-relaxation
    p_new = p.copy()
    omega = 1.5
    for _ in range(50):
        p_new[1:-1,1:-1] = (1-omega)*p[1:-1,1:-1] + omega*(
            (dx2*dz2)/(2*(dx2+dz2)) * (
                (p_new[2:,1:-1] + p_new[:-2,1:-1])/dx2 +
                (p_new[1:-1,2:] + p_new[1:-1,:-2])/dz2 -
                rhs[1:-1,1:-1]
            )
        )
        # Periodic in x
        p_new[0,:] = p_new[-2,:]
        p_new[-1,:] = p_new[1,:]
        
    return p_new

# Time stepping
for n in range(nt):
    # Temporary arrays
    u_temp = u.copy()
    w_temp = w.copy()
    b_temp = b.copy()
    
    # Advection-diffusion for velocity
    u[1:-1,1:-1] = u_temp[1:-1,1:-1] + dt*(
        -u_temp[1:-1,1:-1]*(u_temp[2:,1:-1]-u_temp[:-2,1:-1])/(2*dx)
        -w_temp[1:-1,1:-1]*(u_temp[1:-1,2:]-u_temp[1:-1,:-2])/(2*dz)
        -1/dx*(p[2:,1:-1]-p[:-2,1:-1])/(2*dx)
        +nu*(
            (u_temp[2:,1:-1]-2*u_temp[1:-1,1:-1]+u_temp[:-2,1:-1])/dx2
            +(u_temp[1:-1,2:]-2*u_temp[1:-1,1:-1]+u_temp[1:-1,:-2])/dz2
        )
    )
    
    w[1:-1,1:-1] = w_temp[1:-1,1:-1] + dt*(
        -u_temp[1:-1,1:-1]*(w_temp[2:,1:-1]-w_temp[:-2,1:-1])/(2*dx)
        -w_temp[1:-1,1:-1]*(w_temp[1:-1,2:]-w_temp[1:-1,:-2])/(2*dz)
        -1/dz*(p[1:-1,2:]-p[1:-1,:-2])/(2*dz)
        +nu*(
            (w_temp[2:,1:-1]-2*w_temp[1:-1,1:-1]+w_temp[:-2,1:-1])/dx2
            +(w_temp[1:-1,2:]-2*w_temp[1:-1,1:-1]+w_temp[1:-1,:-2])/dz2
        )
        +b[1:-1,1:-1]
    )
    
    # Buoyancy transport
    b[1:-1,1:-1] = b_temp[1:-1,1:-1] + dt*(
        -u_temp[1:-1,1:-1]*(b_temp[2:,1:-1]-b_temp[:-2,1:-1])/(2*dx)
        -w_temp[1:-1,1:-1]*(b_temp[1:-1,2:]-b_temp[1:-1,:-2])/(2*dz)
        +kappa*(
            (b_temp[2:,1:-1]-2*b_temp[1:-1,1:-1]+b_temp[:-2,1:-1])/dx2
            +(b_temp[1:-1,2:]-2*b_temp[1:-1,1:-1]+b_temp[1:-1,:-2])/dz2
        )
    )
    
    # Pressure correction
    p = pressure_solve(u, w)
    
    # Boundary conditions
    # Top and bottom walls
    u[:,0] = u[:,-1] = 0
    w[:,0] = w[:,-1] = 0
    b[:,0] = Lz
    b[:,-1] = 0
    
    # Periodic in x
    u[0,:] = u[-2,:]
    u[-1,:] = u[1,:]
    w[0,:] = w[-2,:]
    w[-1,:] = w[1,:]
    b[0,:] = b[-2,:]
    b[-1,:] = b[1,:]

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Rayleigh_Benard_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/w_2D_Rayleigh_Benard_Convection.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Rayleigh_Benard_Convection.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/b_2D_Rayleigh_Benard_Convection.npy', b)