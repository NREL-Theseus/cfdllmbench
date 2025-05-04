import numpy as np

# Parameters
nx = 100
nt = 500
dx = 2 * np.pi / nx
dt = 5 / nt
x = np.linspace(0, 2 * np.pi, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Finite difference method (Lax-Friedrichs)
for n in range(nt):
    u_old = u.copy()
    for i in range(nx):
        u[i] = 0.5 * (u_old[(i - 1) % nx] + u_old[(i + 1) % nx]) - \
               0.5 * dt / dx * u_old[i] * (u_old[(i + 1) % nx] - u_old[(i - 1) % nx])

# Save the solution at the final time step
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Nonlinear_Convection.npy', u)