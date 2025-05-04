import numpy as np

def solve_shock_tube():
    # Parameters
    gamma = 1.4
    nx = 200
    nt = 500
    xmin = -1.0
    xmax = 1.0
    t_final = 0.25

    # Grid
    x = np.linspace(xmin, xmax, nx)
    dx = (xmax - xmin) / (nx - 1)
    dt = t_final / nt

    # Initial conditions
    rho = np.where(x < 0, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0, 1.0, 0.1)
    E = p / ((gamma - 1) * rho) + 0.5 * u**2

    # Conservative variables
    U = np.array([rho, rho * u, rho * E])

    # Numerical flux (Lax-Friedrichs)
    def flux(U):
        rho = U[0]
        u = U[1] / rho
        E = U[2] / rho
        p = (gamma - 1) * rho * (E - 0.5 * u**2)
        F = np.array([rho * u, rho * u**2 + p, u * (rho * E + p)])
        return F

    def max_eigenvalue(U):
        rho = U[0]
        u = U[1] / rho
        E = U[2] / rho
        p = (gamma - 1) * rho * (E - 0.5 * u**2)
        a = np.sqrt(gamma * p / rho)
        return np.max(np.abs(u) + a)

    # Time loop
    for n in range(nt):
        F = flux(U)
        max_lambda = max_eigenvalue(U)
        
        F_left = np.zeros_like(F)
        F_right = np.zeros_like(F)
        
        for i in range(U.shape[0]):
            F_left[i, :-1] = 0.5 * (F[i, :-1] + F[i, 1:]) - 0.5 * max_lambda * (U[i, 1:] - U[i, :-1])
            F_right[i, 1:] = 0.5 * (F[i, :-1] + F[i, 1:]) + 0.5 * max_lambda * (U[i, 1:] - U[i, :-1])

        # Update
        for i in range(U.shape[0]):
            U[i, 1:-1] = U[i, 1:-1] - dt/dx * (F_right[i, 1:-1] - F_left[i, 1:-1])

        # Boundary conditions (reflective)
        U[:, 0] = U[:, 1]
        U[1, 0] = -U[1, 1]
        U[:, -1] = U[:, -2]
        U[1, -1] = -U[1, -2]

    # Extract final solution
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1) * rho * (E - 0.5 * u**2)

    # Save the variables
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/rho_1D_Euler_Shock_Tube.npy', rho)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/u_1D_Euler_Shock_Tube.npy', u)
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_1D_Euler_Shock_Tube.npy', p)

solve_shock_tube()