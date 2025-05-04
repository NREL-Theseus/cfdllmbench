import numpy as np

def solve_cfd():
    # Parameters
    nx = 64
    nz = 128
    nt = 200
    dt = 0.1
    nu = 1 / (5 * 10**4)
    D = nu / 1
    
    # Domain
    x = np.linspace(0, 1, nx)
    z = np.linspace(-1, 1, nz)
    X, Z = np.meshgrid(x, z)
    
    # Initial conditions
    u = 0.5 * (1 + np.tanh((Z - 0.5) / 0.1) - np.tanh((Z + 0.5) / 0.1))
    w = 0.01 * np.sin(2 * np.pi * X) * np.exp(-((Z - 0.5)**2 + (Z + 0.5)**2) / 0.01)
    s = u.copy()
    p = np.zeros_like(u)
    
    # Finite difference functions
    def laplacian(f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]) / (dx**2) + \
                           (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1]) / (dz**2)
        return lap

    def advection(u, w, f):
        adv = np.zeros_like(f)
        adv[1:-1, 1:-1] = u[1:-1, 1:-1] * (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * dx) + \
                           w[1:-1, 1:-1] * (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * dz)
        return adv

    def pressure_gradient(p):
        dpdx = np.zeros_like(p)
        dpdz = np.zeros_like(p)
        dpdx[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
        dpdz[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dz)
        return dpdx, dpdz

    def divergence(u, w):
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + (w[2:, 1:-1] - w[:-2, 1:-1]) / (2 * dz)
        return div

    def poisson(p, b):
        p_new = p.copy()
        for _ in range(100):
            p_new[1:-1, 1:-1] = 0.25 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - dx**2 * b[1:-1, 1:-1])
            p = p_new.copy()
        return p

    # Grid spacing
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    
    # Time loop
    for n in range(nt):
        # Solve momentum equations
        u_new = u + dt * (-advection(u, w, u) - pressure_gradient(p)[0] + nu * laplacian(u))
        w_new = w + dt * (-advection(u, w, w) - pressure_gradient(p)[1] + nu * laplacian(w))
        
        # Enforce incompressibility
        b = divergence(u_new, w_new)
        p = poisson(p, b)
        dpdx, dpdz = pressure_gradient(p)
        u_new -= dt * dpdx
        w_new -= dt * dpdz
        
        # Solve tracer equation
        s_new = s + dt * (-advection(u, w, s) + D * laplacian(s))
        
        # Update variables
        u = u_new.copy()
        w = w_new.copy()
        s = s_new.copy()
        
        # Apply periodic boundary conditions
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]

        w[:, 0] = w[:, -2]
        w[:, -1] = w[:, 1]
        w[0, :] = w[-2, :]
        w[-1, :] = w[1, :]

        s[:, 0] = s[:, -2]
        s[:, -1] = s[:, 1]
        s[0, :] = s[-2, :]
        s[-1, :] = s[1, :]

    # Save the final solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_2D_Shear_Flow_With_Tracer.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/w_2D_Shear_Flow_With_Tracer.npy', w)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/s_2D_Shear_Flow_With_Tracer.npy', s)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Shear_Flow_With_Tracer.npy', p)

solve_cfd()