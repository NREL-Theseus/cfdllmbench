import numpy as np

# Parameters
r_inner = 0.5
r_outer = 10.0
nu = 0.005
v_inf = 1.0
t_final = 10.0

# Grid parameters
nr = 50
ntheta = 60
dt = 0.001
r = np.linspace(r_inner, r_outer, nr)
theta = np.linspace(0, 2 * np.pi, ntheta)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

# Initialize arrays
psi = np.zeros((nr, ntheta))
omega = np.zeros((nr, ntheta))

# Initial conditions
# psi[:, :] = 0.0
# omega[:, :] = 0.0

# Boundary conditions
psi[0, :] = 20.0
psi[-1, :] = v_inf * r[-1] * np.sin(theta) + 20.0
omega[0, :] = 0.0  # Will be updated later
omega[-1, :] = 0.0

# Time loop
t = 0.0
while t < t_final:
    # Calculate velocity components
    u_r = np.zeros((nr, ntheta))
    u_theta = np.zeros((nr, ntheta))
    for i in range(nr):
        for j in range(ntheta):
            u_r[i, j] = (psi[i, (j + 1) % ntheta] - psi[i, (j - 1) % ntheta]) / (2 * r[i] * dtheta)
            if i > 0:
                u_theta[i, j] = -(psi[i, j] - psi[i - 1, j]) / dr
            else:
                u_theta[i, j] = -(psi[i+1, j] - psi[i, j]) / dr # Corrected indexing

    # Vorticity transport equation (explicit)
    omega_new = np.zeros((nr, ntheta))
    for i in range(1, nr - 1):
        for j in range(ntheta):
            domega_dt = (
                - u_r[i, j] * (omega[i + 1, j] - omega[i - 1, j]) / (2 * dr)
                - u_theta[i, j] / r[i] * (omega[i, (j + 1) % ntheta] - omega[i, (j - 1) % ntheta]) / (2 * dtheta)
                + nu * ((omega[i + 1, j] - 2 * omega[i, j] + omega[i - 1, j]) / dr**2
                       + (omega[i + 1, j] - omega[i - 1, j]) / (r[i] * 2 * dr)
                       + (omega[i, (j + 1) % ntheta] - 2 * omega[i, j] + omega[i, (j - 1) % ntheta]) / (r[i]**2 * dtheta**2))
            )
            omega_new[i, j] = omega[i, j] + dt * domega_dt

    # Boundary conditions for vorticity
    omega_new[0, :] = 2 * (psi[1, :] - psi[0, :]) / dr**2
    omega_new[-1, :] = 0.0

    # Periodic boundary conditions
    for i in range(nr):
        omega_new[i, :] = omega_new[i, :]

    # Solve Poisson equation for streamfunction
    psi_new = np.zeros((nr, ntheta))
    psi_new[0, :] = 20.0
    psi_new[-1, :] = v_inf * r[-1] * np.sin(theta) + 20.0

    # Iterate to solve the Poisson equation
    for _ in range(50):
        for i in range(1, nr - 1):
            for j in range(ntheta):
                psi_new[i, j] = 0.25 * (
                    psi[i + 1, j] + psi[i - 1, j] + psi[i, (j + 1) % ntheta] + psi[i, (j - 1) % ntheta]
                    + dr**2 * omega_new[i, j]
                )

        psi_new[0, :] = 20.0
        psi_new[-1, :] = v_inf * r[-1] * np.sin(theta) + 20.0

    # Periodic boundary conditions
    for i in range(nr):
        psi_new[i, :] = psi_new[i, :]

    # Update solutions
    omega = omega_new.copy()
    psi = psi_new.copy()

    t += dt

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/psi_Flow_Past_Circular_Cylinder.npy', psi)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/omega_Flow_Past_Circular_Cylinder.npy', omega)