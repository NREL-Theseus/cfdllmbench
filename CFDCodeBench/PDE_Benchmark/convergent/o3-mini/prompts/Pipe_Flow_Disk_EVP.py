#!/usr/bin/env python3
import numpy as np
from scipy.linalg import eig

# Parameters
N = 50             # number of radial grid points
Re = 1e4
kz = 1.0           # axial wavenumber
dr = 1.0/(N-1)     # uniform radial grid spacing
r = np.linspace(0.0, 1.0, N)
w0 = 1.0 - r**2    # laminar background flow

# Total unknowns: u, w, p each defined at N grid points
# Unknown ordering:
#   u: indices 0        ... N-1
#   w: indices N        ... 2*N-1
#   p: indices 2*N      ... 3*N-1
nvars = 3 * N

# We will assemble a generalized eigenvalue problem: A x = s B x,
# where x = [u; w; p] and s is the eigenvalue.
# In the momentum equations, s multiplies u and w (but not p or continuity).
A = np.zeros((nvars, nvars), dtype=complex)
B = np.zeros((nvars, nvars), dtype=complex)

# We'll assemble equations on two sets:
# (I) For interior points (i = 1,...,N-2) we enforce the PDEs.
#     We have three PDEs: continuity, u-momentum, and w-momentum.
# (II) For the boundaries (at r=0 and r=1) we impose BCs.
#
# The order in our assembled system will be:
#   A) Interior equations for i=1,...,N-2 (for each PDE in order: continuity, then u‐momentum, then w‐momentum).
#   B) Boundary conditions at r=0 and r=1 for u, w, and p.
#
# Count of interior points: N_int = N-2.
N_int = N - 2
neq_interior = 3 * N_int
neq_BC = 6   # BC: at r=0: u'(0)=0, w'(0)=0, p(0)=0; at r=1: u(1)=0, w(1)=0, p(1)=0.
total_eq = neq_interior + neq_BC

if total_eq != nvars:
    raise ValueError("Total number of equations does not equal number of unknowns.")

# Helper indices for unknown blocks:
def u_idx(i):
    return i
def w_idx(i):
    return N + i
def p_idx(i):
    return 2 * N + i

row = 0  # equation row counter

# ---------------------------
# (I) Interior equations for i=1,...,N-2
# ---------------------------
for i in range(1, N-1):
    # Use central differences at interior points.
    # Note: i corresponds to the grid index in our array.
    ri = r[i]
    
    # ---------------------------
    # (I-a) Continuity: u_r' + (1/r)*u + i*kz * w = 0
    # Approximate u_r' ~ (u[i+1]-u[i-1])/(2dr)
    # Equation: (u[i+1]-u[i-1])/(2dr) + (1/ri)*u[i] + 1j*kz * w[i] = 0.
    A[row, u_idx(i-1)] += -1.0/(2*dr)
    A[row, u_idx(i+1)] +=  1.0/(2*dr)
    A[row, u_idx(i)]   +=  1.0/ri
    A[row, w_idx(i)]   +=  1j*kz
    # No s-term, so no entry in B.
    row += 1

    # ---------------------------
    # (I-b) u-momentum:
    # Equation: s*u + i*kz*w0*u + dp/dr - (1/Re)[ u'' + (1/r)u' - (1/r^2) u - kz^2 u ] = 0.
    # Approximate:
    #   u'' ~ (u[i+1] - 2*u[i] + u[i-1])/(dr**2)
    #   u'  ~ (u[i+1]-u[i-1])/(2*dr)
    #   dp/dr ~ (p[i+1]-p[i-1])/(2*dr)
    # Write as: s*u[i] appears multiplied by u, so in B set coefficient 1.
    B[row, u_idx(i)] = 1.0
    # Add convection term i*kz*w0:
    A[row, u_idx(i)] += 1j*kz*w0[i]
    # Pressure gradient:
    A[row, p_idx(i-1)] += -1.0/(2*dr)
    A[row, p_idx(i+1)] +=  1.0/(2*dr)
    # Diffusion operator for u:
    coeff_u_ip = (1.0/Re) * (1.0/(dr**2) + 1.0/(2*dr*ri))
    coeff_u_im = (1.0/Re) * (1.0/(dr**2) - 1.0/(2*dr*ri))
    coeff_u_i  = (1.0/Re)*( -2.0/(dr**2) - ( -1.0/(ri**2) - kz**2 ) )
    # Note: the term in brackets is: u'' + (1/r)*u' - (1/r^2)*u - kz^2*u.
    A[row, u_idx(i+1)] += - (1.0/Re)/ (dr**2) - (1.0/Re)*(1.0/(2*dr*ri))
    A[row, u_idx(i-1)] += - (1.0/Re)/ (dr**2) + (1.0/Re)*(1.0/(2*dr*ri))
    A[row, u_idx(i)]   += - (1.0/Re)*(-1.0/(ri**2) - kz**2) + (2.0/(1.0*Re))/(dr**2)  *(-1)  # Adjust sign
    # To be consistent, write the diffusion term as:
    # Diff = (1/Re)[ (u[i+1]-2*u[i]+u[i-1])/(dr**2) + (1/(2dr*ri))*(u[i+1]-u[i-1]) - (1/ri**2 + kz**2)*u[i] ]
    A[row, u_idx(i+1)] += - (1.0/Re)* ( 1.0/(dr**2) + 1.0/(2*dr*ri) )
    A[row, u_idx(i-1)] += - (1.0/Re)* ( 1.0/(dr**2) - 1.0/(2*dr*ri) )
    A[row, u_idx(i)]   += + (1.0/Re)*(2.0/(dr**2)) + (1.0/Re)*(1/ri**2 + kz**2)
    row += 1

    # ---------------------------
    # (I-c) w-momentum:
    # Equation: s*w + i*kz*w0*w - 2*r*u + i*kz*p - (1/Re)[w'' + (1/r)w' - kz**2*w] = 0.
    B[row, w_idx(i)] = 1.0  # s multiplies w
    A[row, w_idx(i)] += 1j*kz*w0[i]
    # Coupling with u: -2*r*u:
    A[row, u_idx(i)] += -2.0 * ri
    # Pressure term: i*kz*p (no derivative)
    A[row, p_idx(i)] += 1j*kz
    # Diffusion for w:
    # Approximate w'' ~ (w[i+1]-2*w[i]+w[i-1])/(dr**2), w' ~ (w[i+1]-w[i-1])/(2dr)
    A[row, w_idx(i+1)] += - (1.0/Re) * (1.0/(dr**2) + 1.0/(2*dr*ri))
    A[row, w_idx(i-1)] += - (1.0/Re) * (1.0/(dr**2) - 1.0/(2*dr*ri))
    A[row, w_idx(i)]   += + (1.0/Re)*(2.0/(dr**2)) + (1.0/Re)*(kz**2)
    row += 1

# ---------------------------
# (II) Boundary conditions
#  At r=0 and r=1, enforce:
#    at r=0: u'(0)=0, w'(0)=0, p(0)=0 (fix pressure gauge)
#    at r=1: u(1)=0, w(1)=0, p(1)=0
# ---------------------------
# BC at r = 0 (i = 0)
i = 0
# (II-a) u'(0)=0 using a forward difference: (-3*u[0] + 4*u[1] - u[2])/(2dr)=0.
A[row, u_idx(0)] += -3.0/(2*dr)
A[row, u_idx(1)] +=  4.0/(2*dr)
A[row, u_idx(2)] += -1.0/(2*dr)
# No s-term => row in B remains 0.
row += 1
# (II-b) w'(0)=0 using forward difference.
A[row, w_idx(0)] += -3.0/(2*dr)
A[row, w_idx(1)] +=  4.0/(2*dr)
A[row, w_idx(2)] += -1.0/(2*dr)
row += 1
# (II-c) p(0)=0
A[row, p_idx(0)] = 1.0
row += 1

# BC at r = 1 (i = N-1)
i = N-1
# (II-d) u(1)=0
A[row, u_idx(N-1)] = 1.0
row += 1
# (II-e) w(1)=0
A[row, w_idx(N-1)] = 1.0
row += 1
# (II-f) p(1)=0
A[row, p_idx(N-1)] = 1.0
row += 1

# Now we have assembled a generalized eigenvalue problem: A x = s B x.
# In our formulation, s only appears multiplying the u and w variables in the momentum equations,
# and we have inserted the identity in B accordingly.
# For the continuity equation and the BC equations, B=0.
# Solve the eigenvalue problem:
evals, evecs = eig(A, B)

# For demonstration, choose the eigenmode with the largest real part of s.
s_index = np.argmax(evals.real)
s_chosen = evals[s_index]

# Extract the eigenvector
x = evecs[:, s_index]
# Normalize (scale so that maximum absolute u is 1)
u_sol = x[0:N]
w_sol = x[N:2*N]
p_sol = x[2*N:3*N]

# Make sure to take the real part if the solution is nearly real.
# (In eigenvalue problems with Fourier modes, the eigenfunctions may be complex.)
u_sol = np.real(u_sol)
w_sol = np.real(w_sol)
p_sol = np.real(p_sol)

# Save the final solution fields as .npy files.
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/u_sol_Pipe_Flow_Disk_EVP.npy', u_sol)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/w_sol_Pipe_Flow_Disk_EVP.npy', w_sol)
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_sol_Pipe_Flow_Disk_EVP.npy', p_sol)