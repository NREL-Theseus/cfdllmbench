import numpy as np

# Grid parameters
nx = 128
nz = 256
dx = 1.0/nx
dz = 2.0/nz
x = np.linspace(0, 1, nx)
z = np.linspace(-1, 1, nz)
X, Z = np.meshgrid(x, z)

# Time parameters
dt = 0.001
t_final = 20
nt = int(t_final/dt)

# Physical parameters
nu = 1/(5e4)  # kinematic viscosity
D = nu/1      # tracer diffusivity

# Initialize fields
u = np.zeros((nz, nx))
w = np.zeros((nz, nx))
s = np.zeros((nz, nx))
p = np.zeros((nz, nx))

# Initial conditions
u = 0.5*(1 + np.tanh((Z-0.5)/0.1) - np.tanh((Z+0.5)/0.1))
w = 0.01*np.sin(2*np.pi*X)*np.exp(-((Z-0.5)**2 + (Z+0.5)**2)/0.1**2)
s = u.copy()

# Helper functions
def periodic_bc(field):
    field[0,:] = field[-2,:]
    field[-1,:] = field[1,:]
    field[:,0] = field[:,-2]
    field[:,-1] = field[:,1]
    return field

def solve_pressure_poisson(u, w, p):
    pn = np.zeros_like(p)
    for _ in range(50):
        pn = p.copy()
        p[1:-1,1:-1] = 0.25*(pn[1:-1,2:] + pn[1:-1,:-2] + pn[2:,1:-1] + pn[:-2,1:-1] - 
                             dx*dz/dt*(
                                 (u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                                 (w[2:,1:-1] - w[:-2,1:-1])/(2*dz)
                             ))
        p = periodic_bc(p)
    return p

# Time stepping
for n in range(nt):
    # Store previous values
    un = u.copy()
    wn = w.copy()
    sn = s.copy()
    
    # Solve pressure
    p = solve_pressure_poisson(un, wn, p)
    
    # Update velocity components
    u[1:-1,1:-1] = (un[1:-1,1:-1] - 
                    dt*un[1:-1,1:-1]*(un[1:-1,2:] - un[1:-1,:-2])/(2*dx) -
                    dt*wn[1:-1,1:-1]*(un[2:,1:-1] - un[:-2,1:-1])/(2*dz) -
                    dt*(p[1:-1,2:] - p[1:-1,:-2])/(2*dx) +
                    nu*dt*(
                        (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 +
                        (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dz**2
                    ))
    
    w[1:-1,1:-1] = (wn[1:-1,1:-1] -
                    dt*un[1:-1,1:-1]*(wn[1:-1,2:] - wn[1:-1,:-2])/(2*dx) -
                    dt*wn[1:-1,1:-1]*(wn[2:,1:-1] - wn[:-2,1:-1])/(2*dz) -
                    dt*(p[2:,1:-1] - p[:-2,1:-1])/(2*dz) +
                    nu*dt*(
                        (wn[1:-1,2:] - 2*wn[1:-1,1:-1] + wn[1:-1,:-2])/dx**2 +
                        (wn[2:,1:-1] - 2*wn[1:-1,1:-1] + wn[:-2,1:-1])/dz**2
                    ))
    
    # Update tracer
    s[1:-1,1:-1] = (sn[1:-1,1:-1] -
                    dt*un[1:-1,1:-1]*(sn[1:-1,2:] - sn[1:-1,:-2])/(2*dx) -
                    dt*wn[1:-1,1:-1]*(sn[2:,1:-1] - sn[:-2,1:-1])/(2*dz) +
                    D*dt*(
                        (sn[1:-1,2:] - 2*sn[1:-1,1:-1] + sn[1:-1,:-2])/dx**2 +
                        (sn[2:,1:-1] - 2*sn[1:-1,1:-1] + sn[:-2,1:-1])/dz**2
                    ))
    
    # Apply boundary conditions
    u = periodic_bc(u)
    w = periodic_bc(w)
    s = periodic_bc(s)

# Save final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)